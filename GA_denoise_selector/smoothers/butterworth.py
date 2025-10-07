"""Zero-phase Butterworth low-pass filtering."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import butter, filtfilt


@dataclass
class ButterworthParams:
    order: int
    cutoff: float  # normalized (0, 0.5)


def make_params(order_idx: int, cutoff_idx: int, order_choices: Sequence[int], cutoff_choices: Sequence[float]) -> ButterworthParams:
    order = order_choices[int(order_idx) % len(order_choices)]
    cutoff = cutoff_choices[int(cutoff_idx) % len(cutoff_choices)]
    return ButterworthParams(order=order, cutoff=cutoff)


def validate_params(params: ButterworthParams) -> bool:
    return 1 <= params.order <= 10 and 0.01 <= params.cutoff < 0.49


def apply(signal: np.ndarray, params: ButterworthParams) -> np.ndarray:
    b, a = butter(params.order, params.cutoff * 2.0, btype="low", analog=False)
    return filtfilt(b, a, signal, axis=-1)


def apply_matrix(matrix: np.ndarray, params: ButterworthParams) -> np.ndarray:
    return np.apply_along_axis(apply, 1, matrix, params)


def complexity(params: ButterworthParams) -> float:
    return float(params.order * 2)
