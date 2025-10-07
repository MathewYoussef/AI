# GA Denoise Selector

Welcome to the genetic-algorithm denoising playbook. This module searches a catalog of classical smoothers to denoise normalized Nostoc reflectance spectra while preserving pigment-rich structure.

## Quick Start
1. Create and activate a virtual environment (Python 3.11+ is recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Edit `config/settings.yaml` to point `normalized_root` at your normalized spectra and tweak GA settings as needed.
4. Launch the GA sweep:
   ```bash
   ./.venv/bin/python orchestrator.py
   ```
5. Review artefacts in `output/` and, if desired, generate plots:
   ```bash
   ./.venv/bin/python generate_post_ga_winner_plots.py
   ```

## Documentation & Workflows
- **Runbook**: `GA_denoise_selector.md` — full setup, configuration reference, fitness details, and troubleshooting.
- **Benchmarking**: `compare_smoothers.py` — compare alternative denoisers with the GA winner.
- **Visuals**: `generate_post_ga_winner_plots.py` — build publication-ready plots for the winning smoother.

## Current Status
The latest high-generation run converged on a Savitzky–Golay smoother (order 4, window 7) that delivers:
- Robust SNR ≈ 50.5 dB
- High-frequency energy reduction ≈ 7.4 %
- Median change fraction ≈ 0.22 %

Re-run the GA with your preferred settings to explore new candidates, or pair the comparison harness with neural denoisers for head-to-head evaluations.
