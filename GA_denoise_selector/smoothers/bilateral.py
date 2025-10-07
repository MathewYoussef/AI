"""Bilateral filtering for 1-D spectra."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from skimage.restoration import denoise_bilateral


@dataclass
class BilateralParams:
    sigma_color: float
    sigma_spatial: float


def make_params(color_idx: int, spatial_idx: int,
                color_choices: Sequence[float], spatial_choices: Sequence[float]) -> BilateralParams:
    sigma_color = color_choices[int(color_idx) % len(color_choices)]
    sigma_spatial = spatial_choices[int(spatial_idx) % len(spatial_choices)]
    return BilateralParams(sigma_color=sigma_color, sigma_spatial=sigma_spatial)


def validate_params(params: BilateralParams) -> bool:
    return params.sigma_color > 0 and params.sigma_spatial > 0


def apply(signal: np.ndarray, params: BilateralParams) -> np.ndarray:
    return denoise_bilateral(signal, sigma_color=params.sigma_color,
                             sigma_spatial=params.sigma_spatial, channel_axis=None)


def apply_matrix(matrix: np.ndarray, params: BilateralParams) -> np.ndarray:
    return np.apply_along_axis(apply, 1, matrix, params)


def complexity(params: BilateralParams) -> float:
    return float(params.sigma_spatial)
