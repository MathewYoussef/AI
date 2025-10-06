#!/usr/bin/env python3
"""Run denoising for all spectra listed in a manifest using a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import SpectraDataset, SpectraDatasetConfig
from src.models.mamba_uv import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV manifest with relative_path column")
    parser.add_argument("--root-dir", required=True, help="Root directory containing spectra")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--output-root", required=True, help="Directory to store _denoised.npy files")
    parser.add_argument("--sequence-length", type=int, default=601, help="Spectral sequence length (default 601)")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    parser.add_argument(
        "--film-features",
        nargs="+",
        default=None,
        help="Conditioning features (default: use checkpoint config if available)",
    )
    parser.add_argument(
        "--film-hidden-dim",
        type=int,
        default=None,
        help="FiLM hidden dimension (default: checkpoint config)",
    )
    parser.add_argument(
        "--dose-features-csv",
        "--dose_features_csv",
        dest="dose_features_csv",
        default=None,
        help="CSV with per-spectrum dose features",
    )
    parser.add_argument(
        "--dose-stats-json",
        "--dose_stats_json",
        dest="dose_stats_json",
        default=None,
        help="JSON file with conditioning mean/std",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=None,
        help="Override model d_model (default: checkpoint config)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=None,
        help="Override model n_layers (default: checkpoint config)",
    )
    parser.add_argument(
        "--d-state",
        type=int,
        default=None,
        help="Override model d_state (default: checkpoint config)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable AMP during inference",
    )
    return parser.parse_args()


def load_checkpoint_config(ckpt_path: Path) -> Dict:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    config = checkpoint.get("config", {})
    state = checkpoint.get("ema_state") or checkpoint.get("model_state")
    if state is None:
        raise ValueError(f"Checkpoint {ckpt_path} does not contain model weights")
    return checkpoint, config


def build_dataset(args: argparse.Namespace, config: Dict) -> SpectraDataset:
    film_features = args.film_features or config.get("film_features") or ["cos_theta"]

    cond_mean: Optional[List[float]] = None
    cond_std: Optional[List[float]] = None
    stats_path = Path(args.dose_stats_json) if args.dose_stats_json else None
    if stats_path and stats_path.exists():
        stats = json.loads(stats_path.read_text())
        cond_mean = stats.get("mean")
        cond_std = stats.get("std")

    cfg = SpectraDatasetConfig(
        sequence_length=args.sequence_length,
        train_dir=args.root_dir,
        noise2noise=False,
        target_wavelengths=None,
        split="train",
        manifest_path=args.manifest,
        dose_features_path=args.dose_features_csv or config.get("dose_features_csv"),
        film_features=tuple(film_features),
        cond_mean=cond_mean,
        cond_std=cond_std,
    )
    return SpectraDataset(cfg)


def build_model_from_checkpoint(
    checkpoint: Dict,
    config: Dict,
    sequence_length: int,
    cond_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    model_kwargs = {
        "sequence_length": sequence_length,
        "in_channels": 3,
        "out_activation": config.get("out_activation", "sigmoid"),
    }
    d_model = args.d_model or config.get("d_model")
    if d_model:
        model_kwargs["d_model"] = int(d_model)
    n_layers = args.n_layers or config.get("n_layers")
    if n_layers:
        model_kwargs["n_layers"] = int(n_layers)
    d_state = args.d_state or config.get("d_state")
    if d_state:
        model_kwargs["d_state"] = int(d_state)
    if cond_dim > 0:
        model_kwargs["geometry_film"] = True
        film_hidden = args.film_hidden_dim or config.get("film_hidden_dim", 64)
        model_kwargs["film_hidden_dim"] = int(film_hidden)
        model_kwargs["cond_dim"] = cond_dim

    model = build_model(config.get("model", "mamba_tiny_uv"), **model_kwargs)
    state = checkpoint.get("ema_state") or checkpoint.get("model_state")
    if state is None:
        raise KeyError(
            f"Checkpoint {args.checkpoint} missing both 'ema_state' and 'model_state' entries"
        )
    state = {
        key.replace("module.", ""): value
        for key, value in state.items()
        if key != "n_averaged"
    }
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint).resolve()
    checkpoint, ckpt_config = load_checkpoint_config(ckpt_path)

    dataset = build_dataset(args, ckpt_config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    cond_dim = dataset.conditioning[0].numel() if dataset.conditioning else 0
    model = build_model_from_checkpoint(
        checkpoint,
        ckpt_config,
        dataset.lambda_norm.shape[0],
        cond_dim,
        args,
        device,
    )

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    idx_offset = 0
    amp_dtype = torch.bfloat16 if args.amp else torch.float16
    autocast_enabled = args.amp and device.type == "cuda"

    with torch.no_grad():
        for noisy, _, cond in loader:
            noisy = noisy.to(device)
            cond_tensor = cond.to(device) if cond_dim > 0 else None
            with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=amp_dtype):
                prediction = model(noisy, cond=cond_tensor) if cond_dim > 0 else model(noisy)
            prediction = prediction.squeeze(1).to(torch.float32).cpu().numpy()

            for i, rel_path in enumerate(dataset.relative_paths[idx_offset: idx_offset + prediction.shape[0]]):
                out_path = output_root / Path(rel_path).with_name(Path(rel_path).stem + "_denoised.npy")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, prediction[i])
            idx_offset += prediction.shape[0]

    print(f"Saved denoised spectra for {len(dataset)} entries to {output_root}")


if __name__ == "__main__":
    main()
