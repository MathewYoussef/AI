"""Total variation denoising."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from skimage.restoration import denoise_tv_chambolle


@dataclass
class TVParams:
    weight: float


def make_params(idx: int, choices: Sequence[float]) -> TVParams:
    weight = choices[int(idx) % len(choices)]
    return TVParams(weight=weight)


def validate_params(params: TVParams) -> bool:
    return params.weight > 0


def apply(signal: np.ndarray, params: TVParams) -> np.ndarray:
    return denoise_tv_chambolle(signal, weight=params.weight, channel_axis=None)


def apply_matrix(matrix: np.ndarray, params: TVParams) -> np.ndarray:
    return np.apply_along_axis(apply, 1, matrix, params)


def complexity(params: TVParams) -> float:
    return float(params.weight)
