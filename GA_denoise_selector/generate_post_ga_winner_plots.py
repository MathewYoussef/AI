#!/usr/bin/env python3
"""Generate plots for the GA-winning smoother."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import orchestrator
import GA_for_smoother as ga

# Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "plots_post_GA_winner"
GA_BEST_JSON = BASE_DIR / "output" / "GA_denoise_best.json"
PIGMENT_CSV = PROJECT_ROOT / "Raw_Normalize" / "plotting_tools" / "cyanobacteria_pigments.csv"

# Plot defaults
PLOT_DPI = 300
DEFAULT_ANGLES = ["12Oclock", "6Oclock"]


def load_spectra_with_names(norm_root: Path,
                            treatments: Iterable[str] | None,
                            angles: Iterable[str] | None,
                            wl_min: float,
                            wl_max: float) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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


def group_indices_by_treatment_angle(file_names: Sequence[str]) -> Dict[Tuple[str, str], List[int]]:
    groups: Dict[Tuple[str, str], List[int]] = {}
    for idx, name in enumerate(file_names):
        parts = name.split("/")
        if len(parts) < 3:
            continue
        treatment = parts[0].replace("treatment_", "")
        angle = parts[2]
        groups.setdefault((treatment, angle), []).append(idx)
    return groups


def percentiles(data: np.ndarray, percentiles: Sequence[float]) -> Dict[float, np.ndarray]:
    return {p: np.percentile(data, p, axis=0) for p in percentiles}


def huber_mean(values: np.ndarray, c: float = 1.345, tol: float = 1e-6, max_iter: int = 100) -> float:
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        return np.nan
    if finite_vals.size == 1:
        return float(finite_vals[0])
    mu = np.median(finite_vals)
    scale = np.median(np.abs(finite_vals - mu)) / 0.6745
    if not np.isfinite(scale) or scale <= 0:
        scale = np.std(finite_vals)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    for _ in range(max_iter):
        residuals = (finite_vals - mu) / scale
        weights = np.ones_like(residuals)
        mask = np.abs(residuals) > c
        weights[mask] = c / np.abs(residuals[mask])
        numerator = np.sum(weights * finite_vals)
        denominator = np.sum(weights)
        if denominator == 0:
            break
        new_mu = numerator / denominator
        if abs(new_mu - mu) < tol:
            mu = new_mu
            break
        mu = new_mu
        mad = np.median(np.abs(finite_vals - mu))
        if mad > 0:
            scale = mad / 0.6745
    return float(mu)


def huber_mean_vector(spectra: np.ndarray) -> np.ndarray:
    return np.array([huber_mean(spectra[:, idx]) for idx in range(spectra.shape[1])], dtype=float)


def parse_ranges(text: str) -> List[Tuple[float, float]]:
    clean = text.replace("–", "-").replace("—", "-")
    comps = [c.strip() for c in clean.split(",") if c.strip()]
    ranges: List[Tuple[float, float]] = []
    for comp in comps:
        match = re.findall(r"\d+(?:\.\d+)?", comp)
        if not match:
            continue
        if len(match) == 1:
            val = float(match[0])
            ranges.append((val, val))
        else:
            low, high = float(match[0]), float(match[1])
            if low > high:
                low, high = high, low
            ranges.append((low, high))
    return ranges


def load_pigment_peaks(csv_path: Path, wl_min: float, wl_max: float) -> List[Tuple[str, float, float]]:
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    peaks: Dict[Tuple[float, float], List[str]] = {}
    for _, row in df.iterrows():
        pigment = str(row.get("Pigment", "Unknown")).strip()
        text = str(row.get("Absorption spectrum", ""))
        for low, high in parse_ranges(text):
            if high < wl_min or low > wl_max:
                continue
            key = (low, high)
            peaks.setdefault(key, []).append(pigment)
    flattened: List[Tuple[str, float, float]] = []
    for (low, high), names in sorted(peaks.items()):
        unique_names = ", ".join(sorted(set(names)))
        flattened.append((unique_names, low, high))
    return flattened


def apply_winning_smoother(config: dict,
                           wavelengths: np.ndarray,
                           spectra: np.ndarray) -> Tuple[np.ndarray, str, Dict[str, object]]:
    with GA_BEST_JSON.open() as fh:
        best_payload = json.load(fh)

    families_cfg = config.get("families")
    active_families = list(families_cfg) if families_cfg else list(ga.AVAILABLE_FAMILIES)

    ga_section = config.get("ga", {})
    exclude = {str(name).strip() for name in ga_section.get("exclude_smoothers", [])}

    smoother_specs = ga.build_smoother_specs(active_families)
    if exclude:
        smoother_specs = [spec for spec in smoother_specs if spec.name not in exclude]
        if not smoother_specs:
            raise SystemExit("❌  All smoothers were excluded via ga.exclude_smoothers")

    best_genome = best_payload["genome"]
    smoother_idx = int(round(best_genome[0]))
    if smoother_idx < 0 or smoother_idx >= len(smoother_specs):
        raise SystemExit("❌  Stored genome references a smoother index outside the active registry")
    spec = smoother_specs[smoother_idx]
    params = ga.decode_parameters(spec, best_genome[1:1 + len(spec.param_options)])
    smoothed = spec.apply(spectra, params, wavelengths)
    return smoothed, spec.name, ga.serialize_params(params)


def plot_median_per_treatment(wavelengths: np.ndarray,
                              smoothed: np.ndarray,
                              groups: Dict[Tuple[str, str], List[int]],
                              treatments: Sequence[str],
                              angles: Sequence[str],
                              out_dir: Path,
                              wl_min: float,
                              wl_max: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(treatments)))

    for idx, treatment in enumerate(treatments):
        indices = []
        for angle in angles:
            indices.extend(groups.get((treatment, angle), []))
        if not indices:
            continue
        data = smoothed[indices]
        stats = percentiles(data, [25, 50, 75])
        ax.plot(wavelengths, stats[50], color=colors[idx], label=f"Treatment {treatment}")
        ax.fill_between(wavelengths, stats[25], stats[75], color=colors[idx], alpha=0.2)

    ax.set_title("Polar-exposure median representative spectra per treatment (post GA)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance (denoised)")
    ax.set_xlim(wl_min, wl_max)
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "polar_exposure_median_spectra.png", dpi=PLOT_DPI)
    plt.close(fig)


def plot_robust_means_by_angle(wavelengths: np.ndarray,
                               smoothed: np.ndarray,
                               groups: Dict[Tuple[str, str], List[int]],
                               treatments: Sequence[str],
                               angles: Sequence[str],
                               out_dir: Path,
                               wl_min: float,
                               wl_max: float,
                               peaks: List[Tuple[str, float, float]] | None = None,
                               suffix: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(angles), figsize=(12, 5), sharey=True, sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    colors = plt.cm.tab10(np.linspace(0, 1, len(treatments)))
    peaks = peaks or []

    for ax, angle in zip(axes, angles):
        for idx, treatment in enumerate(treatments):
            indices = groups.get((treatment, angle), [])
            if not indices:
                continue
            data = smoothed[indices]
            huber = huber_mean_vector(data)
            ax.plot(wavelengths, huber, color=colors[idx], label=f"Treatment {treatment}")

        for j, (name, low, high) in enumerate(peaks):
            if low == high:
                ax.axvline(low, color="k", linestyle="--", linewidth=0.8, alpha=0.45,
                           label="Literature peak" if j == 0 else None)
                ax.annotate(f"{name} ({low:.0f} nm)",
                            xy=(low, ax.get_ylim()[1]),
                            xytext=(1, -10),
                            textcoords="offset points",
                            rotation=90,
                            va="top",
                            fontsize=5,
                            color="k")
            else:
                ax.axvspan(low, high, color="k", alpha=0.15,
                           label="Literature range" if j == 0 else None)
                ax.annotate(f"{name} ({low:.0f}-{high:.0f} nm)",
                            xy=((low + high) / 2, ax.get_ylim()[1]),
                            xytext=(0, -13),
                            textcoords="offset points",
                            rotation=90,
                            ha="center",
                            va="top",
                            fontsize=5,
                            color="k")

        ax.set_title(f"{angle} robust mean spectra (post GA)")
        ax.set_xlabel("Wavelength (nm)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Reflectance (denoised)")
    axes[0].legend(loc="best", frameon=False)
    axes[0].set_xlim(wl_min, wl_max)

    fig.tight_layout()
    fname = "robust_mean_with_peaks" if peaks else "robust_mean_by_angle"
    fig.savefig(out_dir / f"{fname}{suffix}.png", dpi=PLOT_DPI)
    plt.close(fig)


def main() -> None:
    config = orchestrator.load_config()
    norm_root = (BASE_DIR / config.get("normalized_root", "")).resolve()
    treatments = config.get("treatments") or None
    angles = config.get("angles") or DEFAULT_ANGLES
    wl_min = float(config.get("wavelength_min", 300.0))
    wl_max = float(config.get("wavelength_max", 600.0))

    wavelengths, spectra, file_names = load_spectra_with_names(norm_root, treatments, angles, wl_min, wl_max)
    smoothed, smoother_name, params = apply_winning_smoother(config, wavelengths, spectra)

    groups = group_indices_by_treatment_angle(file_names)
    if not treatments:
        treatments = sorted({t for t, _ in groups.keys()})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_median_per_treatment(wavelengths, smoothed, groups, treatments, angles, OUTPUT_DIR, wl_min, wl_max)

    peaks_full = load_pigment_peaks(PIGMENT_CSV, wl_min, wl_max)
    plot_robust_means_by_angle(wavelengths, smoothed, groups, treatments, angles, OUTPUT_DIR, wl_min, wl_max)
    plot_robust_means_by_angle(wavelengths, smoothed, groups, treatments, angles, OUTPUT_DIR, wl_min, wl_max,
                               peaks=peaks_full)

    # 300-490 nm crop for literature peaks
    wl_mask = (wavelengths >= 300.0) & (wavelengths <= 490.0)
    wl_crop = wavelengths[wl_mask]
    smoothed_crop = smoothed[:, wl_mask]
    peaks_crop = load_pigment_peaks(PIGMENT_CSV, 300.0, 490.0)
    plot_robust_means_by_angle(wl_crop, smoothed_crop, groups, treatments, angles, OUTPUT_DIR, 300.0, 490.0,
                               peaks=peaks_crop, suffix="_490_crop")

    # Write a small manifest
    manifest = {
        "smoother": smoother_name,
        "parameters": params,
        "wavelength_min": wl_min,
        "wavelength_max": wl_max,
        "treatments": list(treatments),
        "angles": list(angles),
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(manifest, indent=2))

    print(f"Plots written to {OUTPUT_DIR} using smoother {smoother_name}")


if __name__ == "__main__":
    main()
