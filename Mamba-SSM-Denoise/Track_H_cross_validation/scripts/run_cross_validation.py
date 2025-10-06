#!/usr/bin/env python3
"""Orchestrate sequential cross-validation folds for the Track H denoiser."""

from __future__ import annotations

import argparse
import csv
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List


def append_audit(audit_path: Path, message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} | {message}\n")
    print(f"[audit] {message}")


def run_command(cmd: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed (see {log_path}): {' '.join(cmd)}")


def append_summary(summary_path: Path, row: List[str]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow([
                "fold",
                "train_manifest",
                "val_manifest",
                "test_manifest",
                "checkpoint",
                "train_log",
                "audit_log",
            ])
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-config", default="configs/folds.json")
    parser.add_argument("--folds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--train-dir", default="data/spectra_for_fold")
    parser.add_argument("--val-dir", default="data/spectra_for_fold")
    parser.add_argument("--dose-features-csv", default="data/metadata/dose_features.csv")
    parser.add_argument("--dose-stats-json", default="data/metadata/dose_stats.json")
    parser.add_argument(
        "--film-features",
        nargs="+",
        default=[
            "cos_theta",
            "UVA_total",
            "UVB_total",
            "UVA_over_UVB",
            "P_UVA_mW_cm2",
            "P_UVB_mW_cm2",
            "UVA_norm",
            "UVB_norm",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=600)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Micro-batch accumulation steps for training",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=21)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-min", type=float, default=3e-5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-root", default="logs")
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--denoised-root", default="denoised/folds")
    parser.add_argument("--summary-csv", default="artifacts/cross_validation/fold_summary.csv")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--infer-batch-size", type=int, default=64)
    parser.add_argument("--infer-num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fold_config = Path(args.fold_config).resolve()
    generate_script = Path("scripts/generate_fold_manifests.py").resolve()
    denoise_script = Path("scripts/run_denoise_from_manifest.py").resolve()

    train_dir = Path(args.train_dir).resolve()
    val_dir = Path(args.val_dir).resolve()
    dose_features_csv = Path(args.dose_features_csv).resolve()
    dose_sampling_weights_csv = dose_features_csv.with_name("dose_sampling_weights.csv")
    dose_stats_json = Path(args.dose_stats_json).resolve()

    for fold in args.folds:
        fold_tag = f"fold_{fold:02d}"
        manifest_root = Path("manifests") / fold_tag
        train_manifest = manifest_root / "train_manifest.csv"
        val_manifest = manifest_root / "val_manifest.csv"
        test_manifest = manifest_root / "test_manifest.csv"

        # Generate manifests
        gen_cmd = [
            "python3",
            str(generate_script),
            "--fold",
            str(fold),
            "--config",
            str(fold_config),
            "--input-manifest",
            str(train_dir / "manifest.csv"),
        ]

        # Define directories
        fold_log_dir = Path(args.log_root) / f"Track_H_{fold_tag}"
        fold_log_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = Path(args.checkpoint_root) / f"Track_H_{fold_tag}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        train_log = fold_log_dir / "train.log"
        audit_log = fold_log_dir / "auditing_manifest.log"

        append_audit(audit_log, f"Starting {fold_tag}")
        append_audit(audit_log, "Generating manifests")
        try:
            run_command(gen_cmd, Path(args.log_root) / fold_tag / "generate_manifests.log")
        except Exception as exc:
            append_audit(audit_log, f"Manifest generation failed: {exc}")
            raise
        append_audit(audit_log, f"Generated manifests: {train_manifest}, {val_manifest}, {test_manifest}")

        checkpoint_file = checkpoint_dir / "mamba_tiny_uv_best.pt"
        if args.skip_existing and checkpoint_file.exists():
            append_audit(audit_log, f"Skipping training (checkpoint already exists): {checkpoint_file}")
        else:
            append_audit(audit_log, "Launching training run")
            train_cmd = [
                "python3",
                "-m",
                "src.train",
                "--model",
                "mamba_tiny_uv",
                "--train_dir",
                str(train_dir),
                "--val_dir",
                str(val_dir),
                "--train_manifest",
                str(train_manifest),
                "--val_manifest",
                str(val_manifest),
                "--sequence_length",
                "601",
                "--epochs",
                str(args.epochs),
                "--early_stop_patience",
                str(args.patience),
                "--bs",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--lr_min",
                str(args.lr_min),
                "--weight_decay",
                "1e-4",
                "--noise2noise",
                "--noise2noise_pairwise",
                "--val_noise2noise",
                "--geometry_film",
                "--film_hidden_dim",
                "64",
                "--film_features",
                *args.film_features,
                "--dose_features_csv",
                str(dose_features_csv),
                "--dose_sampling_weights_csv",
                str(dose_sampling_weights_csv),
                "--dose_stats_json",
                str(dose_stats_json),
                "--lambda_weights",
                str(train_dir / "lambda_stats.npz"),
                "--dip_loss",
                "--dip_weight",
                "1.0",
                "--dip_m",
                "6",
                "--dip_window_half_nm",
                "7.0",
                "--dip_min_area",
                "5e-4",
                "--dip_w_area",
                "2.5",
                "--dip_w_equivalent_width",
                "1.5",
                "--dip_w_centroid",
                "2.0",
                "--dip_w_depth",
                "0.3",
                "--dip_underfill_factor",
                "2.0",
                "--dip_detect_sigma_nm",
                "1.0",
                "--baseline",
                "local",
                "--baseline_guard_nm",
                "10.0",
                "--derivative_weight",
                "0.3",
                "--deriv_weight_roi",
                "0.6",
                "--deriv_roi_min",
                "320.0",
                "--deriv_roi_max",
                "500.0",
                "--curvature_weight_roi",
                "0.3",
                "--d_model",
                "384",
                "--n_layers",
                "8",
                "--d_state",
                "16",
                "--amp",
                "--amp_dtype",
                "bf16",
                "--ema",
                "--ema_decay",
                "0.999",
                "--tta_reverse",
                "--cudnn_benchmark",
                "--log_dir",
                str(fold_log_dir),
                "--checkpoint_dir",
                str(checkpoint_dir),
                "--speed_log_json",
                str(Path(args.summary_csv).with_name(f"fold_{fold:02d}_speed.json")),
            ]
            if args.device:
                train_cmd.extend(["--device", args.device])
            if args.grad_accum and args.grad_accum != 1:
                train_cmd.extend(["--grad_accum", str(args.grad_accum)])
            train_cmd.extend(["--audit_log", str(audit_log)])
            try:
                run_command(train_cmd, train_log)
            except Exception as exc:
                append_audit(audit_log, f"Training failed: {exc}")
                raise
            append_audit(audit_log, "Training completed successfully")

        ckpt_path = checkpoint_dir / "mamba_tiny_uv_best.pt"
        if not ckpt_path.exists():
            append_audit(audit_log, "Checkpoint missing - aborting")
            raise FileNotFoundError(f"Checkpoint missing for fold {fold}: {ckpt_path}")
        append_audit(audit_log, f"Checkpoint ready: {ckpt_path}")

        # Run denoising for train/val/test manifests
        denoised_base = Path(args.denoised_root) / fold_tag
        for split, manifest in [("train", train_manifest), ("val", val_manifest), ("test", test_manifest)]:
            out_dir = denoised_base / split
            infer_cmd = [
                "python3",
                str(denoise_script),
                "--manifest",
                str(manifest),
                "--root-dir",
                str(train_dir),
                "--checkpoint",
                str(ckpt_path),
                "--output-root",
                str(out_dir),
                "--sequence-length",
                "601",
                "--batch-size",
                str(args.infer_batch_size),
                "--num-workers",
                str(args.infer_num_workers),
                "--dose_features_csv",
                str(dose_features_csv),
                "--dose_stats_json",
                str(dose_stats_json),
                "--film-features",
                *args.film_features,
                "--amp",
            ]
            if args.device:
                infer_cmd.extend(["--device", args.device])
            append_audit(audit_log, f"Starting inference for {split} -> {out_dir}")
            try:
                run_command(infer_cmd, fold_log_dir / f"inference_{split}.log")
            except Exception as exc:
                append_audit(audit_log, f"Inference failed for {split}: {exc}")
                raise
            append_audit(audit_log, f"Completed inference for {split}")

        append_summary(
            Path(args.summary_csv),
            [
                fold_tag,
                str(train_manifest.resolve()),
                str(val_manifest.resolve()),
                str(test_manifest.resolve()),
                str(ckpt_path.resolve()),
                str(train_log.resolve()),
                str(audit_log.resolve()),
            ],
        )
        append_audit(audit_log, "Fold completed")
        print(f"[fold {fold}] completed")


if __name__ == "__main__":
    main()
