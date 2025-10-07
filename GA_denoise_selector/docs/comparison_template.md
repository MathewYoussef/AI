# Denoiser Comparison Template

This guide captures the structure we will use when presenting future denoisers against the GA-winning Savitzky–Golay smoother. It pairs the quantitative checkpoints with talking points so reviewers can follow **what was compared, how it was measured, and why the result matters**.

## 1. Required Inputs
- `output/smoothed_OPT.csv` – GA baseline (already versioned).
- Candidate denoiser output in the same matrix layout (rows = spectra, columns = wavelengths).
- Access to the raw normalized spectra under `../Raw_Normalize/detection_pipeline/outputs/raw_normalized/` so metrics can be recomputed.
- Config context (`config/settings.yaml`) identifying wavelength bounds, treatments, and angles included in the comparison.

## 2. Run the Evaluation Harness
```bash
./.venv/bin/python compare_smoothers.py \
  path/to/candidate.csv \
  --candidate-label new_method \
  --baseline output/smoothed_OPT.csv \
  --output-dir output/comparisons
```
Outputs:
- JSON summary (`*_vs_ga_winner.json`) containing aggregate metrics (`robust_snr_db`, `high_freq_energy`, `residual_rms`, `curvature_penalty`, `change_fraction`).
- Per-spectrum CSV for statistical tests (paired SNR, high-frequency energy, residual RMS, curvature).

## 3. Statistical Checklist (to be finalised post-candidate)
- **Primary test**: paired bootstrap on `robust_snr_db` and `high_freq_energy` (1000 resamples) to estimate Δ metrics and 95% CIs.
- **Secondary**: non-parametric Wilcoxon signed-rank on per-spectrum deltas.
- **Effect sizes**: report mean delta plus bootstrap CI for each metric.
- **Multiple comparisons**: if more than two denoisers are evaluated, control for FDR (Benjamini–Hochberg) when declaring improvements.

## 4. Visuals to Regenerate for Parity
Use consistent figure styles for every method:
- Median spectra with IQR (`plots_post_GA_winner/polar_exposure_median_spectra.png` analogue).
- Robust means split by angle, with and without pigment overlays.
- Residual histograms or violin plots using the per-spectrum CSV to show distribution shifts.

## 5. Narrative Outline
1. **Setup** – remind reviewers which dataset slice (treatments, angles, wavelength window) is under test.
2. **Baseline recap** – summarise why the GA chose SG (metrics pulled from `GA_denoise_selector.md`).
3. **Candidate description** – short technical blurb: architecture, hyperparameters, runtime considerations.
4. **Metric comparison** – table of GA metrics + deltas + confidence intervals (from JSON + bootstrap notebook).
5. **Visual evidence** – side-by-side figures for medians, robust means, key residual plots; call out pigment-aligned features.
6. **Interpretation** – answer “so what?” in terms of extremophile detection (e.g., improved preservation of UV shoulders). Tie back to statistical significance.
7. **Next actions** – whether to adopt, iterate, or combine methods.

## 6. Artefact Storage Plan
- Place all comparison outputs under `output/comparisons/<method>/` once multiple candidates exist.
- Keep the JSON + per-spectrum CSV for every run; version-control the scripts/notebooks used for bootstrapping so reviewers can rerun audits.

Fill in this template as soon as a candidate smoother stabilises so the final report reads consistently across methods.
