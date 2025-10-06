import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence

from torch.optim.swa_utils import AveragedModel

import numpy as np
import torch
from torch.utils.data import DataLoader

import pandas as pd


def _query_gpu_utilization(device_index: int) -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    try:
        util_value = torch.cuda.utilization(device_index)
        if util_value is not None:
            return float(util_value)
    except (AttributeError, RuntimeError, TypeError, ModuleNotFoundError):
        pass

    if shutil.which("nvidia-smi") is None:
        return None

    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
        "-i",
        str(device_index),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=2)
    except (subprocess.SubprocessError, ValueError):
        return None
    output = result.stdout.strip().splitlines()
    if not output:
        return None
    try:
        return float(output[0])
    except ValueError:
        return None

from src.datasets import SpectraDataset, SpectraDatasetConfig, create_dataloader
from src.losses import (
    CombinedLoss,
    DerivativeLossConfig,
    DipAwareLoss,
    DipLossConfig,
    spectral_derivative,
)
from src.metrics import compute_metrics
from src.models.mamba_uv import build_model


logger = logging.getLogger('train')

def log_info(message: str) -> None:
    logger.info(message)

def log_warning(message: str) -> None:
    logger.warning(message)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mamba-UV for spectral denoising")
    parser.add_argument("--model", default="mamba_tiny_uv", help="Model name from MODEL_ZOO")
    parser.add_argument("--train_dir", required=True, help="Directory with noisy spectra (.npy/.csv)")
    parser.add_argument("--clean_dir", default=None, help="Directory with clean targets (optional)")
    parser.add_argument("--val_dir", default=None, help="Validation directory (matches train format)")
    parser.add_argument(
        "--train_manifest",
        default=None,
        help="Optional manifest CSV to filter the training split",
    )
    parser.add_argument(
        "--val_manifest",
        default=None,
        help="Optional manifest CSV to filter the validation split",
    )
    parser.add_argument(
        "--val_clean_dir", default=None, help="Validation clean directory when available"
    )
    parser.add_argument(
        "--audit_log",
        default=None,
        help="Optional path to append audit log entries",
    )
    parser.add_argument("--sequence_length", "--L", type=int, default=601, dest="sequence_length")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_min", type=float, default=3e-5)
    parser.add_argument("--bs", "--batch_size", type=int, default=128, dest="batch_size")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Prefetch factor per worker (DataLoader)",
    )
    parser.add_argument(
        "--balance_groups",
        action="store_true",
        help="Sample each (treatment, sample) group evenly per iteration",
    )
    parser.add_argument(
        "--persistent_workers",
        action="store_true",
        help="Keep DataLoader workers alive between epochs",
    )
    parser.add_argument(
        "--no_pin_memory",
        action="store_true",
        help="Disable pinned-memory buffers in the DataLoader",
    )
    parser.add_argument("--noise2noise", action="store_true", help="Enable Noise2Noise pairing for training")
    parser.add_argument("--noise2noise_pairwise", action="store_true", help="Use a different replicate as the Noise2Noise target instead of the leave-one-out mean")
    parser.add_argument(
        "--val_noise2noise",
        action="store_true",
        help="Apply Noise2Noise pseudo-targets to the validation set",
    )
    parser.add_argument(
        "--lambda_weights",
        default=None,
        help="Path to wavelength-wise loss weights (.npy or .npz with 'weights')",
    )
    parser.add_argument(
        "--derivative_weight",
        type=float,
        default=0.3,
        help="Weight for derivative loss component",
    )
    parser.add_argument("--derivative_order", type=int, default=1)
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument(
        "--amp_dtype",
        choices=["fp16", "bf16"],
        default="fp16",
        help="Autocast dtype when --amp is enabled",
    )
    parser.add_argument("--log_dir", default="runs")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=10,
        help="epochs without significant SAM improvement before stopping",
    )
    parser.add_argument(
        "--sam_tolerance",
        type=float,
        default=0.05,
        help="Minimum SAM improvement (degrees) to reset patience",
    )
    parser.add_argument("--device", default=None, help="Override device selection (cpu/cuda)")
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 matrix multiply kernels on CUDA",
    )
    parser.add_argument(
        "--float32_matmul_precision",
        choices=["low", "medium", "high"],
        default=None,
        help="Override torch.float32 matmul precision (PyTorch ≥ 2.0)",
    )
    parser.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        help="Enable cuDNN benchmark autotuning",
    )
    parser.add_argument(
        "--out_activation",
        choices=["sigmoid", "clamp", "none"],
        default="sigmoid",
        help="Output activation applied to the model predictions",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout applied inside Mamba blocks",
    )
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--d_state", type=int, default=None)
    parser.add_argument("--d_conv", type=int, default=None)
    parser.add_argument("--expand", type=int, default=None)
    parser.add_argument(
        "--geometry_film",
        action="store_true",
        help="Enable FiLM conditioning on the cosine incident angle",
    )
    parser.add_argument(
        "--film_hidden_dim",
        type=int,
        default=32,
        help="Hidden width for the geometry FiLM MLP when enabled",
    )
    parser.add_argument(
        "--film_dropout",
        type=float,
        default=0.0,
        help="Dropout applied inside the FiLM head",
    )
    parser.add_argument(
        "--film_features",
        nargs="+",
        default=["cos_theta"],
        help="Conditioning features for FiLM (e.g. cos_theta UVA_total UVB_total UVA_over_UVB)",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="Track an exponential moving average of model parameters",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay factor when --ema is enabled",
    )
    parser.add_argument(
        "--tta_reverse",
        action="store_true",
        help="Use forward + reversed spectral order TTA during evaluation",
    )
    parser.add_argument(
        "--dose_features_csv",
        default=None,
        help="CSV containing per-sample dose features (output of Phase A)",
    )
    parser.add_argument(
        "--dose_sampling_weights_csv",
        default=None,
        help="Sampling-weight CSV to flatten the dose distribution (training split)",
    )
    parser.add_argument(
        "--dose_stats_json",
        default=None,
        help="Optional path to read/write conditioning stats (mean/std) for FiLM features",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="Number of micro-batches to accumulate before optimizer.step()",
    )
    parser.add_argument(
        "--speed_log_json",
        default=None,
        help="Optional JSON file to record per-epoch throughput and GPU stats",
    )
    parser.add_argument(
        "--pair_policy_dir",
        default=None,
        help="Optional directory to dump pair-policy statistics (pairwise Noise2Noise only)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile to optimise the model (PyTorch 2.3+)",
    )
    parser.add_argument("--dip_loss", action="store_true", help="Enable dip-aware shape regulariser")
    parser.add_argument("--dip_weight", type=float, default=1.0, help="Global multiplier for dip loss")
    parser.add_argument(
        "--dip_warmup_epochs",
        type=int,
        default=0,
        help="Number of epochs to skip before activating dip loss",
    )
    parser.add_argument("--dip_m", type=int, default=6, help="Maximum dips per spectrum")
    parser.add_argument(
        "--dip_window_half_nm",
        type=float,
        default=5.0,
        help="Half-width of dip windows in nm",
    )
    parser.add_argument(
        "--dip_min_area", type=float, default=1e-5, help="Minimum target area to consider a dip"
    )
    parser.add_argument(
        "--dip_w_area", type=float, default=1.0, help="Weight for dip area mismatch"
    )
    parser.add_argument(
        "--dip_w_area_start",
        type=float,
        default=None,
        help="Optional starting value for dip area weight (linearly ramp to --dip_w_area)",
    )
    parser.add_argument(
        "--dip_w_equivalent_width",
        type=float,
        default=0.0,
        help="Weight for dip equivalent-width mismatch",
    )
    parser.add_argument(
        "--dip_w_centroid", type=float, default=1.0, help="Weight for dip centroid shift"
    )
    parser.add_argument(
        "--dip_w_centroid_start",
        type=float,
        default=None,
        help="Optional starting value for dip centroid weight (linearly ramp to --dip_w_centroid)",
    )
    parser.add_argument(
        "--dip_curriculum_epochs",
        type=int,
        default=0,
        help="Number of epochs to ramp dip area/centroid weights (0 disables ramp)",
    )
    parser.add_argument(
        "--dip_w_depth", type=float, default=0.2, help="Weight for dip depth mismatch"
    )
    parser.add_argument(
        "--dip_underfill_factor",
        type=float,
        default=2.0,
        help="Penalty multiplier for under-estimating dip area",
    )
    parser.add_argument(
        "--dip_detect_sigma_nm",
        type=float,
        default=1.0,
        help="Detection LoG sigma in nm for dip localisation",
    )
    parser.add_argument(
        "--baseline",
        choices=["local", "continuum", "flat"],
        default="continuum",
        help="Baseline used for dip-aware loss",
    )
    parser.add_argument(
        "--baseline_guard_nm",
        type=float,
        default=5.0,
        help="Guard-band width (nm) when using the local baseline",
    )
    parser.add_argument(
        "--dip_w_curvature",
        type=float,
        default=0.0,
        help="Weight for the dip curvature underfill penalty",
    )
    parser.add_argument(
        "--deriv_weight_roi",
        type=float,
        default=0.0,
        help="Additional derivative-matching weight applied inside the ROI",
    )
    parser.add_argument("--deriv_roi_min", type=float, default=320.0)
    parser.add_argument("--deriv_roi_max", type=float, default=500.0)
    parser.add_argument(
        "--curvature_weight_roi",
        type=float,
        default=0.0,
        help="Additional second-derivative curvature guard applied inside the ROI",
    )
    parser.add_argument("--dip_roi_min", type=float, default=320.0)
    parser.add_argument("--dip_roi_max", type=float, default=500.0)
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        help="Optional checkpoint to load weights from before training",
    )
    return parser.parse_args()


def load_wavelength_grid(train_dir: Path, sequence_length: int) -> np.ndarray:
    grid_path = train_dir / "wavelength_grid.npy"
    if grid_path.exists():
        grid = np.load(grid_path).astype(np.float32)
        if grid.shape[0] != sequence_length:
            raise ValueError(
                f"Wavelength grid length {grid.shape[0]} != sequence length {sequence_length}"
            )
        return grid
    start = 300.0
    stop = 600.0
    if sequence_length > 1:
        return np.linspace(start, stop, sequence_length, dtype=np.float32)
    return np.array([start], dtype=np.float32)


def load_lambda_weights(path: Optional[str], expected_len: int) -> Optional[np.ndarray]:
    if path is None:
        return None
    weights_path = Path(path).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Lambda-weight file {weights_path} not found")
    data = np.load(weights_path)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "weights" not in data:
            raise KeyError("Expected 'weights' key inside the weights npz file")
        weights = data["weights"]
    else:
        weights = data
    weights = weights.astype(np.float32)
    if weights.shape[0] != expected_len:
        raise ValueError(
            f"Lambda weights length {weights.shape[0]} does not match sequence length {expected_len}"
        )
    return weights


def prepare_dataloader(
    args: argparse.Namespace,
    split: str,
    batch_size: int,
    shuffle: bool,
    target_wavelengths: Optional[Sequence[float]],
    noise2noise: bool,
    cond_stats: Optional[Dict[str, Sequence[float]]],
) -> Optional[DataLoader]:
    directory = getattr(args, f"{split}_dir")
    if directory is None:
        return None
    clean_dir = args.clean_dir if split == "train" else args.val_clean_dir
    film_features = tuple(args.film_features) if args.geometry_film else ("cos_theta",)
    cond_mean = cond_stats["mean"] if cond_stats is not None else None
    cond_std = cond_stats["std"] if cond_stats is not None else None
    sampling_weights_path = (
        args.dose_sampling_weights_csv if (split == "train" and args.dose_sampling_weights_csv)
        else None
    )
    manifest_override = args.train_manifest if split == "train" else (args.val_manifest if split == "val" else None)
    cfg = SpectraDatasetConfig(
        sequence_length=args.sequence_length,
        train_dir=directory,
        clean_dir=clean_dir,
        noise2noise=noise2noise,
        pairwise_noise2noise=args.noise2noise_pairwise,
        target_wavelengths=target_wavelengths,
        split=split,
        manifest_path=manifest_override,
        dose_features_path=args.dose_features_csv,
        film_features=film_features,
        cond_mean=cond_mean,
        cond_std=cond_std,
        sampling_weights_path=sampling_weights_path,
    )
    return create_dataloader(
        cfg,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=shuffle,
        pin_memory=not args.no_pin_memory,
        persistent_workers=args.persistent_workers,
        balance_groups=args.balance_groups if split == "train" else False,
        prefetch_factor=args.prefetch_factor,
    )


def forward_with_tta(
    module: torch.nn.Module,
    noisy: torch.Tensor,
    cond: Optional[torch.Tensor],
    use_reverse: bool,
) -> torch.Tensor:
    prediction = module(noisy, cond=cond)
    if not use_reverse:
        return prediction
    reversed_input = noisy.flip(-1)
    reversed_prediction = module(reversed_input, cond=cond).flip(-1)
    return 0.5 * (prediction + reversed_prediction)


def compute_conditioning_stats(
    film_features: Sequence[str],
    dose_df: pd.DataFrame,
) -> Dict[str, Sequence[float]]:
    train_df = dose_df[dose_df.get("split", "train") == "train"].copy()
    if train_df.empty:
        raise ValueError("Dose features CSV must contain train split entries to compute stats")

    values: List[List[float]] = []
    for _, row in train_df.iterrows():
        row_values: List[float] = []
        for feature in film_features:
            key = feature.strip().lower()
            if key == "cos_theta":
                row_values.append(SpectraDataset.angle_to_cos(row.get("angle", "")))
            elif key in {"uva_total", "u_total"}:
                row_values.append(float(row.get("UVA_total_mWh_cm2", 0.0)))
            elif key in {"uvb_total", "v_total"}:
                row_values.append(float(row.get("UVB_total_mWh_cm2", 0.0)))
            elif key in {"uva_hours", "uva_hours_h", "exposure_uva_hours"}:
                row_values.append(float(row.get("UVA_hours_h", 0.0)))
            elif key in {"uvb_hours", "uvb_hours_h", "exposure_uvb_hours"}:
                row_values.append(float(row.get("UVB_hours_h", 0.0)))
            elif key in {"p_uva", "p_uva_mw_cm2", "uva_power"}:
                row_values.append(float(row.get("P_UVA_mW_cm2", 0.0)))
            elif key in {"p_uvb", "p_uvb_mw_cm2", "uvb_power"}:
                row_values.append(float(row.get("P_UVB_mW_cm2", 0.0)))
            elif key == "uva_over_uvb":
                value = float(row.get("UVA_over_UVB", 0.0))
                if not np.isfinite(value):
                    value = 0.0
                row_values.append(value)
            elif key == "uva_norm":
                row_values.append(float(row.get("UVA_norm", 0.0)))
            elif key == "uvb_norm":
                row_values.append(float(row.get("UVB_norm", 0.0)))
            elif key == "uva_total_z":
                row_values.append(float(row.get("UVA_total_z", 0.0)))
            elif key == "uvb_total_z":
                row_values.append(float(row.get("UVB_total_z", 0.0)))
            else:
                raise KeyError(f"Unsupported conditioning feature '{feature}' for stats computation")
        values.append(row_values)

    cond_array = np.asarray(values, dtype=np.float32)
    mean = cond_array.mean(axis=0)
    std = cond_array.std(axis=0)
    std[std < 1e-6] = 1.0
    return {"mean": mean.tolist(), "std": std.tolist()}


def configure_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> torch.optim.lr_scheduler._LRScheduler:
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - args.warmup_epochs, 1),
        eta_min=args.lr_min,
    )
    if args.warmup_epochs <= 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[args.warmup_epochs],
    )


def main() -> None:
    args = parse_args()

    if not args.audit_log:
        raise ValueError('--audit_log is required for training runs')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', force=True)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    audit_path = Path(args.audit_log).resolve()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(audit_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(console_handler)
    log_info(f'Audit log path: {audit_path}')


    if args.float32_matmul_precision and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(args.float32_matmul_precision)

    speed_log_path = Path(args.speed_log_json).resolve() if args.speed_log_json else None
    pair_policy_dir = Path(args.pair_policy_dir).resolve() if args.pair_policy_dir else None

    train_path = Path(args.train_dir).resolve()
    wavelength_grid = load_wavelength_grid(train_path, args.sequence_length)
    args.sequence_length = int(wavelength_grid.shape[0])
    target_wavelengths = wavelength_grid.tolist()
    delta_lambda = float(wavelength_grid[1] - wavelength_grid[0]) if wavelength_grid.size > 1 else 1.0

    lambda_weights = load_lambda_weights(args.lambda_weights, args.sequence_length)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log_info(f"Using device: {device}")
    device_index = torch.cuda.current_device() if device.type == "cuda" else None

    cond_stats: Optional[Dict[str, Sequence[float]]] = None
    dose_df: Optional[pd.DataFrame] = None
    if args.dose_features_csv is not None:
        dose_df = pd.read_csv(args.dose_features_csv)

    if args.geometry_film:
        if args.dose_stats_json and Path(args.dose_stats_json).exists():
            with open(args.dose_stats_json, "r", encoding="utf-8") as stats_file:
                cond_stats = json.load(stats_file)
        else:
            if dose_df is None:
                raise ValueError("--dose_features_csv is required when using --geometry_film")
            cond_stats = compute_conditioning_stats(args.film_features, dose_df)
            if args.dose_stats_json:
                stats_path = Path(args.dose_stats_json)
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                with stats_path.open("w", encoding="utf-8") as stats_file:
                    json.dump(
                        {
                            "features": list(args.film_features),
                            "mean": cond_stats["mean"],
                            "std": cond_stats["std"],
                        },
                        stats_file,
                        indent=2,
                    )

    roi_derivative_mask: Optional[torch.Tensor] = None
    roi_curvature_mask: Optional[torch.Tensor] = None
    if args.deriv_weight_roi > 0.0 or args.curvature_weight_roi > 0.0:
        roi_mask = (
            (wavelength_grid >= args.deriv_roi_min)
            & (wavelength_grid <= args.deriv_roi_max)
        ).astype(np.float32)

        def shrink_mask(mask: np.ndarray, order: int) -> np.ndarray:
            shrunk = mask
            for _ in range(order):
                if shrunk.size <= 1:
                    return shrunk[:0]
                shrunk = shrunk[1:] * shrunk[:-1]
            return shrunk

        if args.deriv_weight_roi > 0.0:
            deriv_mask = shrink_mask(roi_mask.copy(), args.derivative_order)
            if deriv_mask.size > 0 and deriv_mask.max() > 0:
                roi_derivative_mask = torch.from_numpy(deriv_mask.astype(np.float32)).to(device)

        if args.curvature_weight_roi > 0.0:
            curvature_mask = shrink_mask(roi_mask.copy(), 2)
            if curvature_mask.size > 0 and curvature_mask.max() > 0:
                roi_curvature_mask = torch.from_numpy(curvature_mask.astype(np.float32)).to(device)
    model_kwargs = {
        "sequence_length": args.sequence_length,
        "in_channels": args.in_channels,
        "dropout": args.dropout,
        "out_activation": args.out_activation,
    }
    if args.d_model is not None:
        model_kwargs["d_model"] = args.d_model
    if args.n_layers is not None:
        model_kwargs["n_layers"] = args.n_layers
    if args.d_state is not None:
        model_kwargs["d_state"] = args.d_state
    if args.d_conv is not None:
        model_kwargs["d_conv"] = args.d_conv
    if args.expand is not None:
        model_kwargs["expand"] = args.expand
    if args.geometry_film:
        model_kwargs["geometry_film"] = True
        model_kwargs["film_hidden_dim"] = args.film_hidden_dim
        model_kwargs["cond_dim"] = len(args.film_features)
        model_kwargs["film_dropout"] = float(args.film_dropout)

    model = build_model(args.model, **model_kwargs)
    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            log_info("[compile] torch.compile enabled")
        except RuntimeError as exc:
            log_warning(f"[compile] torch.compile failed: {exc}. Continuing without compilation.")
    model.to(device)

    amp_dtype = torch.float16
    if args.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16

    if device.type == "cuda":
        if args.allow_tf32 and hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if args.cudnn_benchmark and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True

    ema_model: Optional[AveragedModel] = None
    if args.ema:
        decay = args.ema_decay

        def ema_avg_fn(averaged_param: torch.Tensor, model_param: torch.Tensor, num_averaged: int) -> torch.Tensor:
            if num_averaged == 0:
                return model_param.detach()
            return averaged_param * decay + model_param.detach() * (1.0 - decay)

        ema_model = AveragedModel(model, avg_fn=ema_avg_fn)
        ema_model.to(device)
        ema_model.eval()

    if args.init_checkpoint:
        checkpoint_path = Path(args.init_checkpoint).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Init checkpoint {checkpoint_path} not found")
        log_info(f"Loading initial weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        def _load_into_model(state_dict: Dict[str, torch.Tensor]) -> None:
            model.load_state_dict(state_dict, strict=False)

        ema_state = checkpoint.get("ema_state")
        if ema_state:
            module_state = {
                key.replace("module.", ""): value
                for key, value in ema_state.items()
                if key.startswith("module.")
            }
            _load_into_model(module_state)
            if ema_model is not None:
                ema_model.load_state_dict(ema_state, strict=False)
        else:
            model_state = checkpoint.get("model_state")
            if model_state is None:
                raise KeyError(
                    f"Checkpoint {checkpoint_path} missing both 'ema_state' and 'model_state' entries"
                )
            _load_into_model(model_state)
            if ema_model is not None:
                ema_model.module.load_state_dict(model_state, strict=False)

    train_loader = prepare_dataloader(
        args,
        "train",
        args.batch_size,
        True,
        target_wavelengths,
        args.noise2noise,
        cond_stats,
    )
    if train_loader is None:
        raise ValueError("--train_dir is required")

    if pair_policy_dir:
        if not args.noise2noise_pairwise:
            raise ValueError("--pair_policy_dir requires --noise2noise_pairwise to be enabled")
        pair_policy_dir.mkdir(parents=True, exist_ok=True)
        train_dataset = train_loader.dataset
        if hasattr(train_dataset, "enable_pair_stats"):
            train_dataset.enable_pair_stats(True)
        else:
            raise RuntimeError("Training dataset does not support pair-policy tracking")
    else:
        train_dataset = train_loader.dataset
        if hasattr(train_dataset, "enable_pair_stats"):
            train_dataset.enable_pair_stats(False)

    val_loader = prepare_dataloader(
        args,
        "val",
        args.batch_size,
        False,
        target_wavelengths,
        args.val_noise2noise,
        cond_stats,
    )

    if args.train_manifest:
        log_info(f"Train manifest: {Path(args.train_manifest).resolve()}")
    else:
        log_info("Train manifest: default manifest.csv")
    log_info(f"Training spectra: {len(train_loader.dataset)} | batches per epoch: {len(train_loader)}")
    if args.val_manifest:
        log_info(f"Validation manifest: {Path(args.val_manifest).resolve()}")
    if val_loader is not None:
        log_info(f"Validation spectra: {len(val_loader.dataset)}")
    else:
        log_warning("Validation loader not provided; skipping validation metrics")

    derivative_cfg = DerivativeLossConfig(
        order=args.derivative_order,
        weight=args.derivative_weight,
    )

    criterion = CombinedLoss(
        derivative_cfg=derivative_cfg,
        lambda_weights=lambda_weights,
        delta_lambda=delta_lambda,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = configure_scheduler(optimizer, args)

    dip_loss_module: Optional[DipAwareLoss] = None
    dip_weight = float(args.dip_weight)
    if args.dip_loss:
        lam_tensor = torch.from_numpy(wavelength_grid).to(device)
        dip_cfg = DipLossConfig(
            roi_nm=(float(args.dip_roi_min), float(args.dip_roi_max)),
            m_dips=int(args.dip_m),
            window_half_nm=float(args.dip_window_half_nm),
            min_area=float(args.dip_min_area),
            w_area=float(args.dip_w_area),
            w_equivalent_width=float(args.dip_w_equivalent_width),
            w_centroid=float(args.dip_w_centroid),
            w_depth=float(args.dip_w_depth),
            underfill_factor=float(args.dip_underfill_factor),
            detect_sigma_nm=float(args.dip_detect_sigma_nm),
            lambda_step_nm=float(delta_lambda),
            baseline=args.baseline,
            baseline_guard_nm=float(args.baseline_guard_nm),
            w_curvature=float(args.dip_w_curvature),
        )
        dip_loss_module = DipAwareLoss(lam_tensor, dip_cfg).to(device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.model))
    except ModuleNotFoundError:
        log_warning("TensorBoard not installed; continuing without summaries")

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    best_sam = float("inf")
    epochs_without_improve = 0

    grad_accum = max(1, args.grad_accum)
    global_step = 0

    # Determine curriculum ramp targets
    area_target = float(args.dip_w_area)
    area_start = float(args.dip_w_area_start) if args.dip_w_area_start is not None else area_target
    centroid_target = float(args.dip_w_centroid)
    centroid_start = (
        float(args.dip_w_centroid_start)
        if args.dip_w_centroid_start is not None
        else centroid_target
    )
    ramp_epochs = max(0, int(args.dip_curriculum_epochs))

    speed_records: List[Dict[str, float]] = []
    pair_policy_last_summary: Optional[Dict[str, object]] = None

    for epoch in range(1, args.epochs + 1):
        if pair_policy_dir:
            train_dataset.reset_pair_stats()
        model.train()

        # Update curriculum weights if applicable
        if dip_loss_module is not None and ramp_epochs > 0:
            alpha = min(1.0, max(0, epoch - 1) / ramp_epochs)
            current_area = area_start + (area_target - area_start) * alpha
            current_centroid = centroid_start + (centroid_target - centroid_start) * alpha
            dip_loss_module.cfg.w_area = current_area
            dip_loss_module.cfg.w_centroid = current_centroid
        elif dip_loss_module is not None:
            dip_loss_module.cfg.w_area = area_target
            dip_loss_module.cfg.w_centroid = centroid_target

        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        loss_buffer = 0.0
        dip_buffer = 0.0
        roi_buffer = 0.0
        curvature_buffer = 0.0
        buffer_count = 0

        if speed_log_path:
            epoch_start_time = time.perf_counter()
            samples_processed = 0
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
        else:
            epoch_start_time = None
            samples_processed = 0

        for step, (noisy, target, cond) in enumerate(train_loader):
            noisy = noisy.to(device)
            target = target.to(device)
            cond_tensor = cond.to(device) if args.geometry_film else None

            samples_processed += noisy.size(0)

            with torch.cuda.amp.autocast(enabled=args.amp, dtype=amp_dtype if args.amp else None):
                prediction = model(noisy, cond=cond_tensor) if args.geometry_film else model(noisy)
                base_loss = criterion(prediction, target)
                dip_component = prediction.new_zeros(())
                roi_component = prediction.new_zeros(())
                curvature_component = prediction.new_zeros(())
                if (
                    dip_loss_module is not None
                    and epoch > args.dip_warmup_epochs
                ):
                    dip_component = dip_loss_module(
                        prediction.squeeze(1), target.squeeze(1)
                    )
                    base_loss = base_loss + dip_weight * dip_component
                if roi_derivative_mask is not None and roi_derivative_mask.numel() > 0:
                    pred_deriv = spectral_derivative(
                        prediction, order=args.derivative_order, delta=delta_lambda
                    )
                    target_deriv = spectral_derivative(
                        target, order=args.derivative_order, delta=delta_lambda
                    )
                    roi_diff = torch.abs(pred_deriv - target_deriv)
                    roi_mask_view = roi_derivative_mask.view(1, 1, -1)
                    roi_component = roi_diff.mul(roi_mask_view).mean() * float(
                        args.deriv_weight_roi
                    )
                    base_loss = base_loss + roi_component
                if (
                    roi_curvature_mask is not None
                    and roi_curvature_mask.numel() > 0
                    and args.curvature_weight_roi > 0.0
                ):
                    pred_curv = spectral_derivative(
                        prediction, order=2, delta=delta_lambda
                    ).abs()
                    target_curv = spectral_derivative(
                        target, order=2, delta=delta_lambda
                    ).abs()
                    curv_diff = torch.clamp(target_curv - pred_curv, min=0.0)
                    curv_mask_view = roi_curvature_mask.view(1, 1, -1)
                    curvature_component = curv_diff.mul(curv_mask_view).mean() * float(
                        args.curvature_weight_roi
                    )
                    base_loss = base_loss + curvature_component

            loss_value = base_loss.detach().item()
            epoch_loss += loss_value * noisy.size(0)
            loss_buffer += loss_value
            dip_buffer += dip_component.detach().item() if dip_loss_module is not None else 0.0
            roi_buffer += roi_component.detach().item() if roi_derivative_mask is not None else 0.0
            curvature_buffer += (
                curvature_component.detach().item()
                if roi_curvature_mask is not None and args.curvature_weight_roi > 0.0
                else 0.0
            )
            buffer_count += 1

            scaled_loss = base_loss / grad_accum
            scaler.scale(scaled_loss).backward()

            do_step = ((step + 1) % grad_accum == 0) or (step + 1 == len(train_loader))
            if do_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema_model is not None:
                    ema_model.update_parameters(model)

                if writer is not None and buffer_count > 0:
                    avg_loss = loss_buffer / buffer_count
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    if dip_loss_module is not None:
                        writer.add_scalar(
                            "train/dip_loss",
                            dip_buffer / buffer_count,
                            global_step,
                        )
                    if roi_derivative_mask is not None and roi_derivative_mask.numel() > 0:
                        writer.add_scalar(
                            "train/deriv_roi_loss",
                            roi_buffer / buffer_count,
                            global_step,
                        )
                    if (
                        roi_curvature_mask is not None
                        and roi_curvature_mask.numel() > 0
                        and args.curvature_weight_roi > 0.0
                    ):
                        writer.add_scalar(
                            "train/curvature_roi_loss",
                            curvature_buffer / buffer_count,
                            global_step,
                        )
                loss_buffer = dip_buffer = roi_buffer = curvature_buffer = 0.0
                buffer_count = 0
                global_step += 1

        epoch_loss /= len(train_loader.dataset)
        if speed_log_path and epoch_start_time is not None:
            epoch_duration = time.perf_counter() - epoch_start_time
            if device.type == "cuda":
                torch.cuda.synchronize()
                max_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
            else:
                max_mem_gb = None
            util_pct = _query_gpu_utilization(device_index) if device_index is not None else None
            examples_per_sec = samples_processed / max(epoch_duration, 1e-9)
            speed_records.append(
                {
                    "epoch": epoch,
                    "samples": int(samples_processed),
                    "duration_sec": epoch_duration,
                    "examples_per_sec": examples_per_sec,
                    "max_mem_GB": max_mem_gb,
                    "gpu_util_pct": util_pct,
                }
            )

        current_lr = optimizer.param_groups[0]["lr"]
        log_info(f"Epoch {epoch}/{args.epochs} | train_loss={epoch_loss:.6f} | lr={current_lr:.6e}")

        val_loss = None
        val_metrics = None
        if val_loader is not None and epoch % args.val_every == 0:
            model.eval()
            if ema_model is not None:
                ema_model.eval()
            total_loss = 0.0
            metric_accumulator = {"psnr": 0.0, "sam_deg": 0.0}
            total_samples = 0
            dip_val_total = 0.0
            roi_val_total = 0.0
            curvature_val_total = 0.0
            with torch.no_grad():
                eval_module = ema_model if ema_model is not None else model
                for noisy, target, cond in val_loader:
                    noisy = noisy.to(device)
                    target = target.to(device)
                    cond_tensor = cond.to(device) if args.geometry_film else None
                    prediction = forward_with_tta(
                        eval_module,
                        noisy,
                        cond_tensor,
                        args.tta_reverse,
                    )
                    batch_loss = criterion(prediction, target)
                    dip_component = prediction.new_zeros(())
                    roi_component = prediction.new_zeros(())
                    curvature_component = prediction.new_zeros(())
                    if (
                        dip_loss_module is not None
                        and epoch > args.dip_warmup_epochs
                    ):
                        dip_component = dip_loss_module(
                            prediction.squeeze(1), target.squeeze(1)
                        )
                        batch_loss = batch_loss + dip_weight * dip_component
                    if roi_derivative_mask is not None and roi_derivative_mask.numel() > 0:
                        pred_deriv = spectral_derivative(
                            prediction, order=args.derivative_order, delta=delta_lambda
                        )
                        target_deriv = spectral_derivative(
                            target, order=args.derivative_order, delta=delta_lambda
                        )
                        roi_diff = torch.abs(pred_deriv - target_deriv)
                        roi_mask_view = roi_derivative_mask.view(1, 1, -1)
                        roi_component = roi_diff.mul(roi_mask_view).mean() * float(
                            args.deriv_weight_roi
                        )
                        batch_loss = batch_loss + roi_component
                    if (
                        roi_curvature_mask is not None
                        and roi_curvature_mask.numel() > 0
                        and args.curvature_weight_roi > 0.0
                    ):
                        pred_curv = spectral_derivative(
                            prediction, order=2, delta=delta_lambda
                        ).abs()
                        target_curv = spectral_derivative(
                            target, order=2, delta=delta_lambda
                        ).abs()
                        curv_diff = torch.clamp(target_curv - pred_curv, min=0.0)
                        curv_mask_view = roi_curvature_mask.view(1, 1, -1)
                        curvature_component = curv_diff.mul(curv_mask_view).mean() * float(
                            args.curvature_weight_roi
                        )
                        batch_loss = batch_loss + curvature_component
                    metrics = compute_metrics(prediction, target)

                    total_loss += batch_loss.item() * noisy.size(0)
                    for key in metric_accumulator:
                        metric_accumulator[key] += metrics[key].item() * noisy.size(0)
                    total_samples += noisy.size(0)
                    if dip_loss_module is not None:
                        dip_val_total += dip_component.item() * noisy.size(0)
                    if roi_derivative_mask is not None and roi_derivative_mask.numel() > 0:
                        roi_val_total += roi_component.item() * noisy.size(0)
                    if (
                        roi_curvature_mask is not None
                        and roi_curvature_mask.numel() > 0
                        and args.curvature_weight_roi > 0.0
                    ):
                        curvature_val_total += curvature_component.item() * noisy.size(0)

            val_loss = total_loss / total_samples
            val_metrics = {
                key: value / total_samples for key, value in metric_accumulator.items()
            }
            log_info(f"  val_loss={val_loss:.6f} | PSNR={val_metrics['psnr']:.2f} dB | SAM={val_metrics['sam_deg']:.3f}°")

            if writer is not None:
                writer.add_scalar("val/loss", val_loss, epoch)
                for key, value in val_metrics.items():
                    writer.add_scalar(f"val/{key}", value, epoch)
                if dip_loss_module is not None and epoch > args.dip_warmup_epochs:
                    writer.add_scalar(
                        "val/dip_loss", dip_val_total / total_samples, epoch
                    )
                if roi_derivative_mask is not None and roi_derivative_mask.numel() > 0:
                    writer.add_scalar(
                        "val/deriv_roi_loss", roi_val_total / total_samples, epoch
                    )
                if (
                    roi_curvature_mask is not None
                    and roi_curvature_mask.numel() > 0
                    and args.curvature_weight_roi > 0.0
                ):
                    writer.add_scalar(
                        "val/curvature_roi_loss",
                        curvature_val_total / total_samples,
                        epoch,
                    )

            sam_improvement = best_sam - val_metrics["sam_deg"]
            if sam_improvement > args.sam_tolerance:
                best_sam = val_metrics["sam_deg"]
                epochs_without_improve = 0
                ckpt_path = Path(args.checkpoint_dir) / f"{args.model}_best.pt"
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "ema_state": (ema_model.state_dict() if ema_model is not None else None),
                        "epoch": epoch,
                        "sam_deg": best_sam,
                        "config": vars(args),
                    },
                    ckpt_path,
                )
                log_info(f"  Saved best checkpoint to {ckpt_path}")
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= args.early_stop_patience:
                    log_info("Early stopping triggered due to SAM plateau")
                    break

        if pair_policy_dir:
            pair_summary = train_dataset.pair_stats_summary()
            pair_summary["epoch"] = epoch
            epoch_path = pair_policy_dir / f"epoch_{epoch:03d}.json"
            epoch_path.write_text(json.dumps(pair_summary, indent=2), encoding="utf-8")
            pair_policy_last_summary = pair_summary

            violations = pair_summary.get("violations", 0)
            if violations:
                raise RuntimeError(
                    f"Pair policy violated at epoch {epoch}: {violations} violations"
                )

            coverage_values = pair_summary.get("coverage_values_pct", []) or []
            coverage_median = pair_summary.get("coverage_median_pct")
            if coverage_values:
                coverage_min = min(coverage_values)
                if coverage_median is not None:
                    log_info(f"  pair_policy coverage min={coverage_min:.2f}% | median={coverage_median:.2f}%")
                    if (
                        epoch >= 5
                        and coverage_median < 60.0
                        and args.num_workers <= 1
                    ):
                        raise RuntimeError(
                            f"Pair policy coverage below expectation (median {coverage_median:.2f}% < 60%)"
                        )
                    elif args.num_workers > 1 and coverage_median < 60.0:
                        log_warning("  pair_policy coverage warning: multi-worker loader prevents accurate coverage; relying on dry-run artifact")

            total_pairs = pair_summary.get("total_pairs", 0)
            distance_hist = pair_summary.get("distance_hist", {}) or {}
            if total_pairs > 0:
                zero_count = distance_hist.get("0", 0)
                if not isinstance(zero_count, (int, float)):
                    try:
                        zero_count = float(zero_count)
                    except (TypeError, ValueError):
                        zero_count = 0
                zero_ratio = zero_count / total_pairs
                if zero_ratio > 0.70:
                    raise RuntimeError(
                        f"Pair policy distance collapsed: {zero_ratio*100:.2f}% of pairs at distance 0"
                    )

        scheduler.step()

    if pair_policy_dir and pair_policy_last_summary:
        summary_path = pair_policy_dir.parent / f"{pair_policy_dir.name}_summary.json"
        summary_path.write_text(json.dumps(pair_policy_last_summary, indent=2), encoding="utf-8")

    if speed_log_path and speed_records:
        throughputs = [record["examples_per_sec"] for record in speed_records]
        util_values = [record["gpu_util_pct"] for record in speed_records if record["gpu_util_pct"] is not None]
        memory_values = [record["max_mem_GB"] for record in speed_records if record["max_mem_GB"] is not None]
        summary = {
            "median_examples_per_sec": median(throughputs) if throughputs else None,
            "mean_examples_per_sec": (sum(throughputs) / len(throughputs)) if throughputs else None,
            "median_gpu_util_pct": median(util_values) if util_values else None,
            "max_mem_GB": max(memory_values) if memory_values else None,
            "epochs_logged": len(speed_records),
        }
        log_payload = {"epochs": speed_records, "summary": summary}
        speed_log_path.parent.mkdir(parents=True, exist_ok=True)
        speed_log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
