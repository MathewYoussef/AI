#!/usr/bin/env python3
"""Orchestrator for running the GA denoiser selector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List
import sys

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as exc:
    raise SystemExit("❌  PyYAML is required to load config/settings.yaml") from exc


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "settings.yaml"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise SystemExit(f"❌  Missing configuration file: {CONFIG_PATH}")
    with CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


def load_ga_module():
    import importlib.util

    ga_path = ROOT / "GA_for_smoother.py"
    spec = importlib.util.spec_from_file_location("ga_denoiser_ga", ga_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"❌  Unable to load GA module at {ga_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def discover_csvs(norm_root: Path, treatments: Iterable[str] | None, angles: Iterable[str] | None) -> List[Path]:
    files: List[Path] = []
    if treatments:
        treatment_dirs = [f"treatment_{str(t).strip()}" for t in treatments]
    else:
        treatment_dirs = [p.name for p in sorted(norm_root.glob("treatment_*"))]
    for t_dir_name in treatment_dirs:
        t_dir = norm_root / t_dir_name
        if not t_dir.exists():
            continue
        replicate_dirs = sorted(t_dir.glob("replicate_*"))
        for replicate_dir in replicate_dirs:
            if angles:
                angle_dirs = [replicate_dir / ang for ang in angles]
            else:
                angle_dirs = [p for p in sorted(replicate_dir.iterdir()) if p.is_dir()]
            for angle_dir in angle_dirs:
                csv_path = angle_dir / "normalized.csv"
                if csv_path.exists():
                    files.append(csv_path)
    return files


def load_spectra(norm_root: Path,
                 treatments: Iterable[str] | None,
                 angles: Iterable[str] | None,
                 wl_min: float,
                 wl_max: float) -> tuple[np.ndarray, np.ndarray]:
    csv_files = discover_csvs(norm_root, treatments, angles)
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
    return wavelengths, spectra


def main() -> None:
    config = load_config()
    norm_root = (ROOT / config.get("normalized_root", "")).resolve()
    if not norm_root.exists():
        raise SystemExit(f"❌  Normalized root not found: {norm_root}")

    treatments = config.get("treatments") or None
    angles = config.get("angles") or None
    wl_min = float(config.get("wavelength_min", 300.0))
    wl_max = float(config.get("wavelength_max", 600.0))
    output_dir = (ROOT / config.get("output_dir", "output")).resolve()
    families = config.get("families")
    ga_section = config.get("ga", {})

    wavelengths, spectra = load_spectra(norm_root, treatments, angles, wl_min, wl_max)
    ga_module = load_ga_module()
    result = ga_module.run_ga(spectra, wavelengths, output_dir, families=families, ga_params=ga_section)

    print(json.dumps({
        "output_dir": str(output_dir),
        "best_fitness": result.fitness,
        "best_genome": list(map(float, result.best_genome)),
        "metrics": result.metrics,
        "smoother": result.smoother_name,
        "parameters": result.parameters,
    }, indent=2))


if __name__ == "__main__":
    main()
