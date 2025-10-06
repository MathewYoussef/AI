"""Utility functions for dark/white calibration of raw spectrometer captures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class SummaryRow:
    """Single entry from ``summary_index.csv`` describing a capture."""

    path: Path
    file_type: str
    treatment: str
    replicate: str
    angle: Optional[str]
    white_panel_type: Optional[str]


def _resolve_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (root / path)


def load_summary(summary_csv: Path, root: Optional[Path] = None) -> List[SummaryRow]:
    """Parse ``summary_index.csv`` into in-memory records."""

    summary_csv = Path(summary_csv)
    if root is None:
        root = summary_csv.parent
    df = pd.read_csv(summary_csv)

    required = {"path", "file_type", "treatment_number", "replicate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"summary_index.csv missing columns: {missing}")

    rows: List[SummaryRow] = []
    for _, row in df.iterrows():
        path = _resolve_path(root, str(row["path"]))
        file_type = str(row["file_type"]).lower()
        treatment = str(row["treatment_number"]).split(".")[0]
        replicate = str(row["replicate"]).strip()
        angle = row.get("angle_clock")
        angle = str(angle).strip() if pd.notna(angle) else None
        white_panel_type = row.get("white_panel_type")
        white_panel_type = str(white_panel_type).strip() if pd.notna(white_panel_type) else None
        rows.append(
            SummaryRow(
                path=path,
                file_type=file_type,
                treatment=treatment,
                replicate=replicate,
                angle=angle,
                white_panel_type=white_panel_type,
            )
        )
    return rows


def read_spectrum_txt(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a raw spectrometer ``.txt`` file into wavelength & intensity arrays."""

    wavelengths: List[float] = []
    intensities: List[float] = []
    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                wavelength = float(parts[0])
                intensity = float(parts[1])
            except ValueError:
                continue
            wavelengths.append(wavelength)
            intensities.append(intensity)

    if not wavelengths:
        raise ValueError(f"No numeric data parsed from {path}")

    wl = np.asarray(wavelengths, dtype=float)
    vals = np.asarray(intensities, dtype=float)
    order = np.argsort(wl)
    return wl[order], vals[order]


def average_scans(files: Iterable[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Average one or more raw scans interpolated to a common grid."""

    file_list = list(files)
    if not file_list:
        raise ValueError("No files provided for averaging")

    base_wl, base_vals = read_spectrum_txt(file_list[0])
    if len(file_list) == 1:
        return base_wl, base_vals

    stack = [base_vals]
    for current in file_list[1:]:
        wl, vals = read_spectrum_txt(current)
        stack.append(np.interp(base_wl, wl, vals))
    return base_wl, np.mean(np.vstack(stack), axis=0)


def build_dark_reference(rows: Sequence[SummaryRow]) -> Tuple[np.ndarray, np.ndarray]:
    """Average all rows labelled as dark references."""

    dark_files = [row.path for row in rows if row.file_type == "dark_ref"]
    if not dark_files:
        raise ValueError("No dark references found in summary")
    return average_scans(dark_files)


def build_white_reference(rows: Sequence[SummaryRow], treatment: str) -> Tuple[np.ndarray, np.ndarray]:
    """Average white references for a specific treatment, prioritising 6% panels."""

    white_rows = [row for row in rows if row.file_type == "white_ref" and row.treatment == treatment]
    if not white_rows:
        raise ValueError(f"No white references available for treatment {treatment}")

    whites_6pct1 = [row.path for row in white_rows if row.white_panel_type and "6pct1" in row.white_panel_type.lower()]
    whites_6pct2 = [row.path for row in white_rows if row.white_panel_type and "6pct2" in row.white_panel_type.lower()]

    wl_ref: Optional[np.ndarray] = None
    vals_ref: Optional[np.ndarray] = None

    if whites_6pct1:
        wl_ref, vals_ref = average_scans(whites_6pct1)
    if whites_6pct2:
        wl2, vals2 = average_scans(whites_6pct2)
        if wl_ref is None:
            wl_ref, vals_ref = wl2, vals2
        else:
            vals_interp = np.interp(wl_ref, wl2, vals2)
            vals_ref = 0.5 * (vals_ref + vals_interp)

    if wl_ref is None or vals_ref is None:
        raise ValueError(f"No 6% white references for treatment {treatment}")
    return wl_ref, vals_ref


def crop_spectrum(wavelengths: np.ndarray, values: np.ndarray, minimum: float, maximum: float) -> Tuple[np.ndarray, np.ndarray]:
    """Restrict a spectrum to the inclusive interval ``[minimum, maximum]``."""

    mask = (wavelengths >= minimum) & (wavelengths <= maximum)
    return wavelengths[mask], values[mask]


def ensure_monotonic(grid: np.ndarray) -> np.ndarray:
    if np.any(np.diff(grid) <= 0):
        raise ValueError("Target wavelength grid must be strictly increasing")
    return grid


def make_wavelength_grid(lambda_min: float, lambda_max: float, lambda_step: float) -> np.ndarray:
    """Create an inclusive wavelength grid."""

    if lambda_step <= 0:
        raise ValueError("lambda_step must be positive")
    points = int(np.round((lambda_max - lambda_min) / lambda_step))
    grid = lambda_min + lambda_step * np.arange(points + 1)
    return ensure_monotonic(grid.astype(np.float32))


def dark_white_calibrate(sample: np.ndarray, dark: np.ndarray, white: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Apply (sample - dark) / (white - dark) with numerical guards.

    Parameters
    ----------
    sample, dark, white:
        Arrays on a shared wavelength grid.
    eps:
        Minimum denominator magnitude to avoid division by zero.
    """

    sample = np.asarray(sample, dtype=float)
    dark = np.asarray(dark, dtype=float)
    white = np.asarray(white, dtype=float)

    if sample.shape != dark.shape or sample.shape != white.shape:
        raise ValueError("sample, dark, white must share shape")

    numerator = sample - dark
    denominator = white - dark
    small = np.abs(denominator) < eps
    denominator = denominator.copy()
    denominator[small] = np.where(denominator[small] >= 0, eps, -eps)

    calibrated = numerator / denominator
    calibrated = np.nan_to_num(calibrated, nan=0.0, posinf=1.0, neginf=0.0)
    return calibrated.astype(np.float32, copy=False)
