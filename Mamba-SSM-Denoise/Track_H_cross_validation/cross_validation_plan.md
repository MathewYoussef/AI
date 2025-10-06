# Cross-Validation Plan

This document records the 5-fold train/val/test rotations for the Track H denoiser
and explains how to generate fold-specific manifests.

## Sample Rotations (per treatment)

| Fold | treatment_1 | treatment_2 | treatment_3 | treatment_4 | treatment_5 |
| ---- | ----------- | ----------- | ----------- | ----------- | ----------- |
| **Train** |
| 1 | sample_A, sample_B, sample_C | sample_A, sample_C, sample_D | sample_B, sample_F, sample_G | sample_C, sample_D, sample_E | sample_A, sample_C, sample_G |
| 2 | sample_B, sample_C, sample_E | sample_C, sample_D, sample_F | sample_F, sample_G, sample_I | sample_D, sample_E, sample_I | sample_C, sample_G, sample_H |
| 3 | sample_C, sample_E, sample_H | sample_D, sample_F, sample_H | sample_G, sample_I, sample_J | sample_E, sample_I, sample_J | sample_G, sample_H, sample_J |
| 4 | sample_E, sample_H, sample_A | sample_F, sample_H, sample_A | sample_I, sample_J, sample_B | sample_I, sample_J, sample_C | sample_H, sample_J, sample_A |
| 5 | sample_H, sample_A, sample_B | sample_H, sample_A, sample_C | sample_J, sample_B, sample_F | sample_J, sample_C, sample_D | sample_J, sample_A, sample_C |
| **Validation** |
| 1 | sample_E | sample_F | sample_I | sample_I | sample_H |
| 2 | sample_H | sample_H | sample_J | sample_J | sample_J |
| 3 | sample_A | sample_A | sample_B | sample_C | sample_A |
| 4 | sample_B | sample_C | sample_F | sample_D | sample_C |
| 5 | sample_C | sample_D | sample_G | sample_E | sample_G |
| **Test** |
| 1 | sample_H | sample_H | sample_J | sample_J | sample_J |
| 2 | sample_A | sample_A | sample_B | sample_C | sample_A |
| 3 | sample_B | sample_C | sample_F | sample_D | sample_C |
| 4 | sample_C | sample_D | sample_G | sample_E | sample_G |
| 5 | sample_E | sample_F | sample_I | sample_I | sample_H |

Each fold trains on three samples per treatment (900 spectra/treatment), validates on
one sample (300 spectra), and tests on the remaining sample (300 spectra). After
five folds every sample has appeared exactly once in each role.

## Generating Manifests

```
# e.g. fold 1
python3 scripts/generate_fold_manifests.py --fold 1
```

This writes CSVs under `manifests/fold_01/`:

- `train_manifest.csv`
- `val_manifest.csv`
- `test_manifest.csv`

Repeat with `--fold 2` … `--fold 5` to produce the full set.

## Training Command Template

```
python3 -m src.train \
  --model mamba_tiny_uv \
  --train_dir data/spectra_for_fold \
  --val_dir data/spectra_for_fold \
  --train_manifest manifests/fold_01/train_manifest.csv \
  --val_manifest manifests/fold_01/val_manifest.csv \
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
  --log_dir logs/Track_H_fold_01 --checkpoint_dir checkpoints/Track_H_fold_01 \
  --audit_log logs/Track_H_fold_01/auditing_manifest.log
```

Adjust `fold_01` to the desired fold. If 600 overflows GPU memory, reduce `--bs` accordingly.

To run all five folds sequentially (training → denoising → logging) use:

```
python3 scripts/run_cross_validation.py
```

## Inference + QA

After training each fold, denoise the assigned train/val/test manifests and archive
metrics. Example placeholder commands:

```
python scripts/export_denoised_csv.py \
  --split fold_01_train \
  --manifest-name train_manifest.csv \
  --split fold_01_val \
  --manifest-name val_manifest.csv \
  --split fold_01_test \
  --manifest-name test_manifest.csv \
  --outfile denoised/fold_01/combined.csv

python scripts/plot_treatment_averages.py \
  --split fold_01_val \
  --raw-root data/spectra_for_fold \
  --denoised-root "denoised/fold_01" \
  --manifest-name val_manifest.csv \
  --outfile plots/fold_01_val_averages.png
```

(Replace with the actual batch inference workflow once wired up.)

## Metric Tracking

For each fold, append validation/test metrics to a CSV such as
`artifacts/fold_metrics_summary.csv` with columns:

```
fold,split,treatment,psnr_db,sam_deg,dip_centroid_mae_nm,dip_area_mape_pct
```

This makes it easy to summarise mean ± standard deviation across folds before
training the final production checkpoint on all samples.

> Best practice: defer ROI metric comparisons until the fold finishes training and inference. Hold hyperparameters constant across folds, then compute aggregate statistics once all five folds complete. Use those aggregates to justify the final full-data retraining instead of selecting an individual fold wholesale.

> Reminder: once the five folds are complete and you retrain on the merged dataset, include `treatment_6` when you denoise the full corpus so the control spectra ship with the final deliverable.
