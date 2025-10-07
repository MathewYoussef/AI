"""Smoothing spline denoiser."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.interpolate import UnivariateSpline


@dataclass
class SplineParams:
    smooth_factor: float


def make_params(idx: int, choices: Sequence[float]) -> SplineParams:
    value = choices[int(idx) % len(choices)]
    return SplineParams(smooth_factor=value)


def validate_params(params: SplineParams) -> bool:
    return params.smooth_factor >= 0


def apply(signal: np.ndarray, params: SplineParams, wavelengths: np.ndarray | None = None) -> np.ndarray:
    if wavelengths is None:
        wavelengths = np.arange(len(signal))
    spline = UnivariateSpline(wavelengths, signal, s=params.smooth_factor * len(signal))
    return spline(wavelengths)


def apply_matrix(matrix: np.ndarray, params: SplineParams, wavelengths: np.ndarray | None = None) -> np.ndarray:
    if wavelengths is None:
        wavelengths = np.arange(matrix.shape[1])
    return np.vstack([apply(row, params, wavelengths=wavelengths) for row in matrix])


def complexity(params: SplineParams) -> float:
    return float(np.log10(params.smooth_factor + 1))
