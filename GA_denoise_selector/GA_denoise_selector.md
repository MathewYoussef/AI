# GA Denoise Selector

## Overview
The GA denoise selector searches a catalog of classical smoothers to denoise normalized Nostoc reflectance spectra (300–600 nm) while preserving pigment-specific structure. It consumes the `Raw_Normalize/detection_pipeline/outputs/raw_normalized/` directory produced by the upstream pipeline, evaluates each candidate with a multi-term fitness function, and stores the best-performing smoother plus diagnostics under `output/` for downstream analysis.

Key components:
- `orchestrator.py` — loads configuration, assembles spectra, runs the GA, and emits artefacts.
- `GA_for_smoother.py` — defines the smoother registry, GA machinery, and metrics.
- `smoothers/` — parameter builders, validators, and apply routines for each denoiser family.
- `compare_smoothers.py` — benchmarking harness for candidate vs. GA winner comparisons.
- `generate_post_ga_winner_plots.py` — optional visual report generator for the winning smoother.

## Quick Start
1. **Install Python 3.11+** (the project is currently tested with 3.13).
2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # macOS/Linux
   # Windows PowerShell
   # .\.venv\Scripts\Activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure the run** by editing `config/settings.yaml` (see the reference below). Make sure `normalized_root` points at the normalized spectra directory.
5. **Launch the GA**:
   ```bash
   ./.venv/bin/python orchestrator.py
   ```
   On long runs (e.g., 200+ generations with GP/PCA enabled) consider using `nohup` or a screen/tmux session so the process can finish uninterrupted.
6. **Review artefacts** in `output/` and (optionally) regenerate plots:
   ```bash
   ./.venv/bin/python generate_post_ga_winner_plots.py
   ```

## Configuration Reference (`config/settings.yaml`)
| Key | Description |
| --- | --- |
| `normalized_root` | Absolute or relative path to normalized spectra. Each spectrum must provide `wavelength_nm` and `normalized` columns. |
| `output_dir` | Where the GA writes artefacts (`smoothed_OPT.csv`, `GA_denoise_best.json`, plots). Relative paths resolve under the project root. |
| `wavelength_min`, `wavelength_max` | Restrict the wavelength window in nanometres. Defaults: 300–600. |
| `treatments`, `angles` | Optional filters (lists). Empty lists include every treatment/angle discovered in `normalized_root`. |
| `families` | Optional subset of wavelet families to consider. Leave empty to use `smoothers.AVAILABLE_FAMILIES`. |
| `ga.num_generations` | Total generations to run (including restarts). Increase for broader search. |
| `ga.sol_per_pop` | Number of solutions per population. Larger populations explore more, but raise runtime. |
| `ga.num_parents_mating` | Tournament-selected parents used for crossover. Defaults to half the population if omitted. |
| `ga.keep_parents` | Elite solutions carried across restarts. |
| `ga.random_seed` | Deterministic seed for reproducibility. Adjust to explore different trajectories. |
| `ga.exclude_smoothers` | Optional list of smoother names to skip (e.g., `gaussian_process`, `pca`) when you need faster iterations. |

### Runtime Tips
- Gaussian Process and PCA smoothers dominate runtime. Exclude them during development, then re-enable for publication-grade sweeps.
- The GA records fitness every generation. Inspect `output/plots/ga_fitness_curve.png` to verify whether the run plateaued or kept improving.
- Restarts trigger after 10 stagnant generations, retaining elites and re-sampling the remainder of the population.

## Metrics and Fitness
Each candidate smoother is scored with the following metrics (computed across all spectra):
- **`robust_snr`** — Median/MAD signal-to-noise ratio (dB) using the raw spectrum as the signal and the raw–smoothed residual as noise.
- **`high_freq_energy`** — Fraction of FFT energy in the highest half of the frequency spectrum (lower is smoother).
- **`residual_rms`** — Mean RMS of the residual.
- **`curvature_penalty`** — Mean absolute second derivative, discouraging over-smoothing.
- **`change_fraction`** — Median absolute change relative to the raw signal (guards against identity and destructive solutions).
- **`complexity`** — Heuristic cost per smoother (e.g., window width, wavelet level).

The fitness function emphasises SNR improvements and high-frequency suppression while bounding the acceptable change window:
```
fitness = 2·robust_snr
          − 2·high_freq_energy − 0.8·residual_rms − 0.5·curvature_penalty − 0.3·complexity
          + 5000·max(raw_high_freq − high_freq_energy, 0)
          − 20000·max(0.002 − change_fraction, 0)
          − 20000·max(change_fraction − 0.05, 0)
```
- `change_fraction` penalties discourage both “no-op” (identity) and overly aggressive smoothers.
- The high-frequency gain bonus rewards spectra that meaningfully suppress noise above the raw baseline.

## Outputs
After each GA run you should see:
- `output/GA_denoise_best.json` — genome, smoother name, parameters, metrics, and effective GA hyperparameters.
- `output/smoothed_OPT.csv` — smoothed spectra matrix (rows = spectra, columns = wavelengths in ascending order).
- `output/plots/ga_fitness_curve.png` — best fitness per generation (stitching together restarts).
- `output/comparisons/` — any JSON/CSV comparisons generated via `compare_smoothers.py`.
- `plots_post_GA_winner/` — optional visual report (median spectra, pigment overlays) generated by `generate_post_ga_winner_plots.py`.

## Comparison Workflow
Use the harness to contrast alternative denoisers with the GA winner:
```bash
./.venv/bin/python compare_smoothers.py \
  output/smoothed_OPT.csv \
  --candidate path/to/candidate.csv \
  --candidate-label new_method \
  --baseline-label ga_winner \
  --output-dir output/comparisons
```
The tool re-computes all GA metrics for both matrices, writes a JSON summary (“baseline”, “candidate”, “delta”), and produces a per-spectrum CSV with paired statistics (`robust_snr_db`, `high_freq_energy`, `residual_rms`, `curvature_penalty`, `change_fraction`). Use these outputs to build statistical tests (bootstrap, Wilcoxon) and figures.

## Current GA Winner (config: 240 generations, sol_per_pop=32, GP/PCA excluded)
- **Smoother**: Savitzky–Golay (`order=4`, `window=7`).
- **Metrics**: `robust_snr` 50.52 dB, `high_freq_energy` 1.87×10⁻³, `residual_rms` 7.26×10⁻⁴, `curvature_penalty` 2.00×10⁻³, `change_fraction` 2.22×10⁻³, `complexity` 7.0.
- **High-frequency reduction**: 7.4% relative to the raw spectra.
- **Median absolute change**: 0.22% of the raw signal magnitude.

Re-run `orchestrator.py` after modifying GA settings (e.g., enabling GP/PCA or increasing `num_generations`). The artefacts mentioned above update in-place.

## Troubleshooting
- **Missing packages**: Ensure you activated `.venv` and installed `requirements.txt`.
- **GA run times out**: Exclude heavy smoothers via `ga.exclude_smoothers`, reduce `num_generations`, or run the command in a persistent session (`nohup`, `tmux`).
- **Matplotlib cache errors**: Set `MPLCONFIGDIR` to a writable location before running plotting scripts (e.g., `export MPLCONFIGDIR=$PWD/.mplcache`).
- **Genome mismatch in plotting helper**: Always regenerate plots after the GA finishes; the helper respects `ga.exclude_smoothers` and stored parameters.

## Suggested Next Steps
1. Tune `ga.*` parameters to find the desired compute/accuracy trade-off. Watch `ga_fitness_curve.png` for additional plateaus.
2. Feed candidate smoothers (e.g., neural models) through `compare_smoothers.py` using the GA winner or SG baselines.
3. Extend the documentation under `docs/` with experiment-specific narratives using `docs/comparison_template.md`.

With the above tooling and documentation, a new team member can reproduce the GA run, interpret the metrics, and benchmark alternative denoisers without additional context.
