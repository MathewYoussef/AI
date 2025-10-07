"""Savitzky–Golay smoother utilities."""

from dataclasses import dataclass
import numpy as np
from scipy.signal import savgol_filter


@dataclass
class SGParams:
    order: int
    window: int


def make_params(order: int, window: int) -> SGParams:
    return SGParams(order=int(round(order)), window=int(round(window)))


def validate_params(params: SGParams) -> bool:
    return params.window % 2 == 1 and params.window > params.order and params.order >= 1


def apply(signal: np.ndarray, params: SGParams) -> np.ndarray:
    return savgol_filter(signal, window_length=params.window, polyorder=params.order)


def apply_matrix(matrix: np.ndarray, params: SGParams) -> np.ndarray:
    return np.apply_along_axis(apply, 1, matrix, params)


def complexity(params: SGParams) -> float:
    return float(params.window)
