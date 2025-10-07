#!/usr/bin/env python3
"""Benchmark denoised spectra against the GA-winning Savitzky–Golay smoother.

This utility normalises the comparison workflow so future denoisers can be
measured with the same metrics, plots, and per-spectrum statistics that the GA
optimiser uses. It outputs a JSON summary plus a CSV with paired metrics to
facilitate statistical testing (e.g., bootstrap, Wilcoxon) once multiple
methods are available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

import orchestrator

try:
    import yaml
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit("❌  PyYAML is required. Install it with `pip install pyyaml`.") from exc

ROOT = Path(__file__).resolve().parent
DEFAULT_BASELINE = ROOT / "output" / "smoothed_OPT.csv"
DEFAULT_OUT_DIR = ROOT / "output" / "comparisons"
DEFAULT_CONFIG = ROOT / "config" / "settings.yaml"


def load_config(config_path: Path) -> dict:
    orchestrator.CONFIG_PATH = config_path
    return orchestrator.load_config()


def load_spectra_with_names(norm_root: Path,
                            treatments: Iterable[str] | None,
                            angles: Iterable[str] | None,
                            wl_min: float,
                            wl_max: float) -> tuple[np.ndarray, np.ndarray, List[str]]:
    csv_files = orchestrator.discover_csvs(norm_root, treatments, angles)
    if not csv_files:
        raise SystemExit(f"❌  No normalized spectra found under {norm_root}")

    frames = []
    for fp in csv_files:
        df = pd.read_csv(fp)
        if {"wavelength_nm", "normalized"} - set(df.columns):
            raise RuntimeError(f"Unexpected columns in {fp}")
        tmp = df.rename(columns={"wavelength_nm": "wavelength", "normalized": "refl"})
        tmp["file_name"] = str(fp.relative_to(norm_root))
        frames.append(tmp)

    frame = pd.concat(frames, ignore_index=True)
    spec_mat = (frame.pivot_table(index="file_name",
                                  columns="wavelength",
                                  values="refl")
                .sort_index(axis=1))
    spec_mat = (spec_mat
                .interpolate(axis=1, limit_direction="both")
                .dropna(axis=0, how="any")
                .dropna(axis=1, how="any"))

    wavelengths = spec_mat.columns.to_numpy(dtype=float)
    if wl_min >= wl_max:
        raise ValueError("wavelength_min must be < wavelength_max")
    mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    if not mask.any():
        raise ValueError("No wavelengths remain after applying bounds")

    wavelengths = wavelengths[mask]
    spectra = spec_mat.to_numpy(dtype=float)[:, mask]
    names = list(spec_mat.index)
    return wavelengths, spectra, names


def load_smoothed_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise SystemExit(f"❌  Smoothed CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"❌  Smoothed CSV {path} is empty")
    wavelengths = df.columns.to_numpy(dtype=float)
    matrix = df.to_numpy(dtype=float)
    return wavelengths, matrix


def ensure_alignment(reference_wavelengths: np.ndarray,
                     candidate_wavelengths: np.ndarray,
                     tolerance: float = 1e-6) -> None:
    if reference_wavelengths.shape != candidate_wavelengths.shape:
        raise SystemExit("❌  Wavelength grids differ between datasets")
    if not np.allclose(reference_wavelengths, candidate_wavelengths, atol=tolerance):
        raise SystemExit("❌  Wavelength grids are misaligned; resample before comparing")


def per_spectrum_metrics(raw: np.ndarray,
                         smoothed: np.ndarray,
                         wavelengths: np.ndarray) -> Dict[str, np.ndarray]:
    residual = raw - smoothed
    change = np.median(np.abs(residual), axis=1)
    signal = np.maximum(np.median(np.abs(raw), axis=1), 1e-9)
    noise = np.median(np.abs(residual), axis=1) / 0.6745
    noise_floor = np.maximum(1e-3 * signal, 1e-9)
    noise = np.maximum(noise, noise_floor)
    ratio = np.maximum(signal / noise, 1e-9)
    snr_db = 20.0 * np.log10(ratio)

    centered = smoothed - smoothed.mean(axis=1, keepdims=True)
    if centered.shape[1] > 1:
        delta = np.median(np.diff(wavelengths))
        fft_vals = np.fft.rfft(centered, axis=1)
        freqs = np.fft.rfftfreq(centered.shape[1], d=delta)
        if freqs.size:
            cutoff = freqs.max() * 0.5
            mask = freqs >= cutoff
            hf = np.sum(np.abs(fft_vals[:, mask]) ** 2, axis=1)
            total = np.sum(np.abs(fft_vals) ** 2, axis=1) + 1e-9
            hf_ratio = hf / total
        else:
            hf_ratio = np.zeros(centered.shape[0])
    else:
        hf_ratio = np.zeros(centered.shape[0])

    rms = np.sqrt(np.mean(residual ** 2, axis=1))
    if smoothed.shape[1] >= 3:
        curvature = np.mean(np.abs(np.diff(smoothed, n=2, axis=1)), axis=1)
    else:
        curvature = np.zeros(smoothed.shape[0])

    return {
        "robust_snr_db": snr_db,
        "high_freq_energy": hf_ratio,
        "residual_rms": rms,
        "curvature_penalty": curvature,
        "change_fraction": change / (signal + 1e-9),
    }


def aggregate_metrics(per_spec: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {key: float(np.mean(values)) for key, values in per_spec.items()}


def build_per_spec_table(names: Sequence[str], metrics: Dict[str, np.ndarray]) -> pd.DataFrame:
    data = {"sample": list(names)}
    data.update({key: metrics[key] for key in metrics})
    return pd.DataFrame(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a candidate denoiser to the GA-winning smoother")
    parser.add_argument("candidate", type=Path, help="CSV of smoothed spectra to evaluate")
    parser.add_argument("--candidate-label", default=None, help="Friendly label for the candidate (defaults to file stem)")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE, help="Baseline smoothed CSV (defaults to GA winner)")
    parser.add_argument("--baseline-label", default="ga_winner", help="Friendly name for the baseline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="GA config file used to source raw spectra")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR, help="Where comparison artefacts will be written")
    parser.add_argument("--treatments", nargs="*", help="Subset of treatments to load (defaults to config)")
    parser.add_argument("--angles", nargs="*", help="Subset of angles to load (defaults to config)")
    parser.add_argument("--wl-min", type=float, default=None, help="Minimum wavelength override")
    parser.add_argument("--wl-max", type=float, default=None, help="Maximum wavelength override")
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    norm_root = (ROOT / config.get("normalized_root", "")).resolve()
    treatments = args.treatments or config.get("treatments") or None
    angles = args.angles or config.get("angles") or None
    wl_min = args.wl_min if args.wl_min is not None else float(config.get("wavelength_min", 300.0))
    wl_max = args.wl_max if args.wl_max is not None else float(config.get("wavelength_max", 600.0))

    wavelengths_raw, raw_spectra, names = load_spectra_with_names(norm_root, treatments, angles, wl_min, wl_max)

    wavelengths_base, baseline = load_smoothed_matrix(args.baseline.resolve())
    wavelengths_cand, candidate = load_smoothed_matrix(args.candidate.resolve())

    ensure_alignment(wavelengths_raw, wavelengths_base)
    ensure_alignment(wavelengths_raw, wavelengths_cand)

    if baseline.shape != raw_spectra.shape:
        raise SystemExit("❌  Baseline smoothed matrix does not match raw spectra shape")
    if candidate.shape != raw_spectra.shape:
        raise SystemExit("❌  Candidate smoothed matrix does not match raw spectra shape")

    baseline_per_spec = per_spectrum_metrics(raw_spectra, baseline, wavelengths_raw)
    candidate_per_spec = per_spectrum_metrics(raw_spectra, candidate, wavelengths_raw)

    baseline_summary = aggregate_metrics(baseline_per_spec)
    candidate_summary = aggregate_metrics(candidate_per_spec)
    deltas = {key: candidate_summary[key] - baseline_summary[key] for key in baseline_summary}

    comparison_dir = args.output_dir.resolve()
    comparison_dir.mkdir(parents=True, exist_ok=True)

    candidate_label = args.candidate_label or args.candidate.stem
    summary_path = comparison_dir / f"{candidate_label}_vs_{args.baseline_label}.json"
    per_spec_path = comparison_dir / f"{candidate_label}_vs_{args.baseline_label}_per_spectrum.csv"

    summary_payload = {
        "baseline_label": args.baseline_label,
        "candidate_label": candidate_label,
        "normalized_root": str(norm_root),
        "wavelength_min": wl_min,
        "wavelength_max": wl_max,
        "num_spectra": int(raw_spectra.shape[0]),
        "metrics": {
            "baseline": baseline_summary,
            "candidate": candidate_summary,
            "delta": deltas,
        },
        "input_files": {
            "baseline": str(args.baseline.resolve()),
            "candidate": str(args.candidate.resolve()),
        },
    }

    summary_path.write_text(json.dumps(summary_payload, indent=2))

    delta_table = build_per_spec_table(names, {
        "baseline_robust_snr_db": baseline_per_spec["robust_snr_db"],
        "candidate_robust_snr_db": candidate_per_spec["robust_snr_db"],
        "baseline_high_freq_energy": baseline_per_spec["high_freq_energy"],
        "candidate_high_freq_energy": candidate_per_spec["high_freq_energy"],
        "baseline_residual_rms": baseline_per_spec["residual_rms"],
        "candidate_residual_rms": candidate_per_spec["residual_rms"],
        "baseline_curvature_penalty": baseline_per_spec["curvature_penalty"],
        "candidate_curvature_penalty": candidate_per_spec["curvature_penalty"],
        "baseline_change_fraction": baseline_per_spec["change_fraction"],
        "candidate_change_fraction": candidate_per_spec["change_fraction"],
    })
    delta_table["delta_robust_snr_db"] = delta_table["candidate_robust_snr_db"] - delta_table["baseline_robust_snr_db"]
    delta_table["delta_high_freq_energy"] = delta_table["candidate_high_freq_energy"] - delta_table["baseline_high_freq_energy"]
    delta_table["delta_residual_rms"] = delta_table["candidate_residual_rms"] - delta_table["baseline_residual_rms"]
    delta_table["delta_curvature_penalty"] = delta_table["candidate_curvature_penalty"] - delta_table["baseline_curvature_penalty"]
    delta_table["delta_change_fraction"] = delta_table["candidate_change_fraction"] - delta_table["baseline_change_fraction"]
    delta_table.to_csv(per_spec_path, index=False)

    print("✅  Comparison complete")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
