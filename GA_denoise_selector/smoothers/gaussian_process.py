"""Gaussian Process regression denoiser."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


@dataclass
class GPParams:
    length_scale: float
    noise_level: float


def make_params(ls_idx: int, noise_idx: int, ls_choices: Sequence[float], noise_choices: Sequence[float]) -> GPParams:
    length_scale = ls_choices[int(ls_idx) % len(ls_choices)]
    noise_level = noise_choices[int(noise_idx) % len(noise_choices)]
    return GPParams(length_scale=length_scale, noise_level=noise_level)


def validate_params(params: GPParams) -> bool:
    return params.length_scale > 0 and params.noise_level >= 0


def apply(signal: np.ndarray, params: GPParams, wavelengths: np.ndarray | None = None) -> np.ndarray:
    if wavelengths is None:
        wavelengths = np.arange(len(signal))[:, None]
    else:
        wavelengths = wavelengths[:, None]
    kernel = Matern(length_scale=params.length_scale, nu=1.5) + WhiteKernel(noise_level=params.noise_level)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    gp.fit(wavelengths, signal)
    return gp.predict(wavelengths)


def apply_matrix(matrix: np.ndarray, params: GPParams, wavelengths: np.ndarray | None = None) -> np.ndarray:
    if wavelengths is None:
        wavelengths = np.arange(matrix.shape[1])
    return np.vstack([apply(row, params, wavelengths=wavelengths) for row in matrix])


def complexity(params: GPParams) -> float:
    return float(np.log(params.length_scale + 1) + params.noise_level)
