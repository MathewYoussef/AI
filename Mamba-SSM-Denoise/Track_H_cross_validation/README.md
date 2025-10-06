# Cross-Validation Runbook

This guide explains how to execute the five-fold Track H denoiser experiment, where to find the outputs, and how to recover from interruptions.

---
## Environment Setup
1. Create a Python 3.10 environment (e.g. `conda create -n spectra python=3.10` then `conda activate spectra`).
2. Install PyTorch with CUDA support that matches your driver. For example, on Linux with CUDA 12.1:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   See https://pytorch.org/get-started/locally/ for other platforms.
3. Install the project dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure `data/spectra_for_fold` is present (treatments 1–6) and `lambda_stats.npz` lives in that directory.

### Hardware Notes
- The automated cross-validation run is sized for a 40 GB GPU (A100 class).
- With smaller cards (≥24 GB) you can lower `--batch-size` via `scripts/run_cross_validation.py --batch-size ...`.
- Expect the full 5-fold pass to occupy ~2 TB disk for denoised outputs and checkpoints; keep the `Mamba-SSM-Denoise` repo on a fast SSD.

---
## 1. Overview
- We train five folds sequentially; each fold uses 3 samples/treatment for training, 1 for validation, 1 for test.
- All steps (manifest generation → training → denoising → logging) are automated, with mandatory audit logs written to `logs/Track_H_fold_##/auditing_manifest.log`.
- Per-fold outputs:
  - `manifests/fold_##/train|val|test_manifest.csv`
  - `logs/Track_H_fold_##/train.log` (stdout/stderr) and `auditing_manifest.log`
  - `checkpoints/Track_H_fold_##/mamba_tiny_uv_best.pt`
  - `denoised/folds/fold_##/{train,val,test}` → `_denoised.npy`
  - metrics summary row in `artifacts/cross_validation/fold_summary.csv`

---
## 2. Prerequisites
1. **Environment**: activate the `spectra` Conda env (or equivalent) with PyTorch, `mamba-ssm`, and repo requirements installed.
   ```bash
   source ~/miniconda/bin/activate spectra
   cd /data/spectra-denoise  # adjust path as needed
   ```
2. **Training data**: ensure `data/spectra_for_fold` contains every spectrum (treatments 1–6, all samples/replicates) and `lambda_stats.npz` is computed for that directory.
3. **Metadata**: `data/metadata/dose_features.csv`, `dose_sampling_weights.csv`, and `dose_stats.json` must be present; they drive FiLM conditioning. The sampling-weights sheet now buckets every spectrum by normalized exposure (quintiles of `UVA_norm`, `UVB_norm`) and records the result as `(uva_bin, uvb_bin)` with unit weights.
4. **GPU resources**: the automation assumes a 40 GB GPU. If you have less memory, reduce `--batch-size` when launching.

---
## 3. One-off preparation
Generate the manifests once for reference (the automation regenerates them per fold, but this acts as a sanity check).
```bash
for fold in 1 2 3 4 5; do
  python3 scripts/generate_fold_manifests.py --fold $fold
done
```
Verify counts (`train=4500`, `val=1500`, `test=1500` per fold).

---
## 4. Launch all folds
Run the orchestrator to process folds **1 → 5** sequentially:
```bash
python3 scripts/run_cross_validation.py
```
Key command-line options:
- `--device cuda:0` (or similar) to pin the GPU
- `--skip-existing` to skip any fold whose checkpoint already exists
- `--batch-size`, `--epochs`, `--patience`, etc., override defaults if needed
- `--folds 3 4 5` to run a subset
All training runs use an audit log; `src/train.py` enforces `--audit_log`, so the orchestrator handles it automatically.

---
## 5. Where logs end up
Per fold (example: fold 02):
| Artifact | Path | Notes |
| --- | --- | --- |
| Training stdout/stderr | `logs/Track_H_fold_02/train.log` | full command + per-epoch output |
| Audit timeline | `logs/Track_H_fold_02/auditing_manifest.log` | timestamped events (manifest generation, training, inference, completion) |
| Checkpoint | `checkpoints/Track_H_fold_02/mamba_tiny_uv_best.pt` | EMA weights preferred; raw weights in same file |
| Denoised spectra | `denoised/folds/fold_02/{train,val,test}` | `_denoised.npy` for each manifest entry |
| Summary row | `artifacts/cross_validation/fold_summary.csv` | includes audit-log and checkpoint locations |

Audit log sample:
```
2026-03-15T19:42:03 | Starting fold_02
2026-03-15T19:42:03 | Generating manifests
2026-03-15T19:42:05 | Generated manifests: manifests/fold_02/train_manifest.csv, ...
2026-03-15T19:42:05 | Launching training run
...
2026-03-15T22:14:49 | Training completed successfully
2026-03-15T22:14:49 | Checkpoint ready: checkpoints/Track_H_fold_02/mamba_tiny_uv_best.pt
2026-03-15T22:14:49 | Starting inference for val -> denoised/folds/fold_02/val
...
2026-03-15T22:18:10 | Fold completed
```

---
## 6. Resume / Skip behaviour
- To resume after a failure, re-run `scripts/run_cross_validation.py`. Use `--skip-existing` to avoid retraining folds with valid checkpoints.
- If you want to re-run only one fold (e.g., fold 04) with fresh logs:
  ```bash
  rm -rf logs/Track_H_fold_04 checkpoints/Track_H_fold_04 denoised/folds/fold_04
  python3 scripts/run_cross_validation.py --folds 4
  ```
- Training failure logs appear in `train.log` and the audit file. Fix the issue, delete the partially written checkpoint, and rerun the fold.

---
## 7. After all folds finish
1. Inspect `artifacts/cross_validation/fold_summary.csv` for paths and to confirm all folds succeeded.
2. Aggregate metrics (PSNR/SAM/dip stats) using the denoised outputs or training logs.
3. Retrain a final model on all samples (no hold-outs) once you’ve reviewed the cross-validation results.
4. Consider archiving the per-fold audit logs alongside metrics for future reference.

### ROI metric evaluation & model selection
- Run ROI metric notebooks/scripts only after each fold reaches its best checkpoint; the automation already denoises train/val/test once training finishes (`scripts/run_cross_validation.py` handles this sequentially).
- Keep hyperparameters fixed for all folds. Tweaking mid-run leaks information between folds and undermines the cross-validation estimate.
- Summarise ROI metrics across folds using mean ± standard deviation. This captures performance stability instead of cherry-picking the easiest split.
- Use the cross-validation review to validate settings, then train one production model on the full dataset. Shipping the single “best” fold leaves 20 % of the data unused and is not representative.

---
## 8. Manual training (single fold)
If you need to run a specific fold manually (e.g., fold 03) outside the automation:
```bash
python3 -m src.train \ 
  --model mamba_tiny_uv \ 
  --train_dir data/spectra_for_fold \ 
  --val_dir data/spectra_for_fold \ 
  --train_manifest manifests/fold_03/train_manifest.csv \ 
  --val_manifest manifests/fold_03/val_manifest.csv \ 
  --sequence_length 601 \ 
  --epochs 200 \ 
  --early_stop_patience 21 \ 
  --bs 600 \ 
  --lr 3e-4 \ 
  --lr_min 3e-5 \ 
  --weight_decay 1e-4 \ 
  --noise2noise --noise2noise_pairwise --val_noise2noise \ 
  --geometry_film --film_hidden_dim 64 \ 
  --film_features cos_theta UVA_total UVB_total UVA_over_UVB P_UVA_mW_cm2 P_UVB_mW_cm2 UVA_norm UVB_norm \ 
  --dose_features_csv data/metadata/dose_features.csv \ 
  --dose_sampling_weights_csv data/metadata/dose_sampling_weights.csv \ 
  --dose_stats_json data/metadata/dose_stats.json \ 
  --lambda_weights data/spectra_for_fold/lambda_stats.npz \ 
  --dip_loss --dip_weight 1.0 --dip_m 6 \ 
  --dip_window_half_nm 7.0 --dip_min_area 5e-4 \ 
  --dip_w_area 2.5 --dip_w_equivalent_width 1.5 --dip_w_centroid 2.0 --dip_w_depth 0.3 \ 
  --dip_underfill_factor 2.0 --dip_detect_sigma_nm 1.0 --baseline local --baseline_guard_nm 10.0 \ 
  --derivative_weight 0.3 --deriv_weight_roi 0.6 --deriv_roi_min 320.0 --deriv_roi_max 500.0 \ 
  --curvature_weight_roi 0.3 \ 
  --d_model 384 --n_layers 8 --d_state 16 \ 
  --amp --amp_dtype bf16 --ema --ema_decay 0.999 --tta_reverse --cudnn_benchmark \ 
  --log_dir logs/Track_H_fold_03 \ 
  --checkpoint_dir checkpoints/Track_H_fold_03 \ 
  --speed_log_json artifacts/cross_validation/fold_03_speed.json \ 
  --audit_log logs/Track_H_fold_03/auditing_manifest.log
```
(Adjust `fold_03` to the desired fold.)

---
## 9. Post-training denoising only
To regenerate denoised spectra for a completed fold:
```bash
python3 scripts/run_denoise_from_manifest.py \ 
  --manifest manifests/fold_03/val_manifest.csv \ 
  --root-dir data/spectra_for_fold \ 
  --checkpoint checkpoints/Track_H_fold_03/mamba_tiny_uv_best.pt \
  --output-root denoised/folds/fold_03/val \
  --sequence-length 601 \
  --batch-size 64 \
  --dose-features-csv data/metadata/dose_features.csv \
  --dose-stats-json data/metadata/dose_stats.json \
  --film-features cos_theta UVA_total UVB_total UVA_over_UVB P_UVA_mW_cm2 P_UVB_mW_cm2 UVA_norm UVB_norm \
  --amp
```

---
## 10. Troubleshooting
- **OOM during training**: reduce `--batch-size` (e.g., 512 or 384). The scheduler will still run for 200 epochs.
- **Gradient explosion**: check logs for `nan` in loss; re-run with a smaller LR or enable gradient clipping inside `src/train.py` (not currently active).
- **Missing spectra**: ensure the manifests point to actual files. The generator raises `FileNotFoundError` if a path references a non-existent sample.
- **Audit log missing entries**: all stages write to the audit file; if it stops mid-run, look at `train.log` or the inference log for the failure reason.
- **Resuming mid-fold**: delete the incomplete checkpoint to force a clean rerun, or add `--skip-existing` to jump over completed folds.

---
Happy training! Keep the audit logs with the metrics summary for an airtight provenance trail.
