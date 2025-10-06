# Metadata Layout

- `dose_features.csv` — per-spectrum UV exposure features used for FiLM conditioning.
- `dose_sampling_weights.csv` — sampling weights and `(uva_bin, uvb_bin)` exposure labels for **every** spectrum in `spectra_for_fold`. The bins are computed from quintiles of the normalized dose columns (`UVA_norm`, `UVB_norm`), and the weights are currently uniform (`1.0`).
- `dose_stats.json` — global mean/std for conditioning features, consumed by the model when FiLM is enabled.

If you change the dataset, regenerate the weights with `UVA_norm`/`UVB_norm` quantiles so audits stay aligned with the physics.
