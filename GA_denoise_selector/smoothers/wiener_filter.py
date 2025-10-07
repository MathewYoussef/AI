"""Wiener filtering."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import wiener


@dataclass
class WienerParams:
    window: int
    noise_power: float


def make_params(window_idx: int, noise_idx: int, window_choices: Sequence[int], noise_choices: Sequence[float]) -> WienerParams:
    window = window_choices[int(window_idx) % len(window_choices)]
    noise = noise_choices[int(noise_idx) % len(noise_choices)]
    return WienerParams(window=window, noise_power=noise)


def validate_params(params: WienerParams) -> bool:
    return params.window >= 3 and params.window % 2 == 1 and params.noise_power > 0


def apply(signal: np.ndarray, params: WienerParams) -> np.ndarray:
    return wiener(signal, mysize=params.window, noise=params.noise_power)


def apply_matrix(matrix: np.ndarray, params: WienerParams) -> np.ndarray:
    return np.apply_along_axis(apply, 1, matrix, params)


def complexity(params: WienerParams) -> float:
    return float(params.window)
