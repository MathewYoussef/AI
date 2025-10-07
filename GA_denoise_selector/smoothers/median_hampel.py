"""Median and Hampel filters for spike removal."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import medfilt


@dataclass
class MedianHampelParams:
    kernel_size: int
    hampel_window: int
    hampel_sigma: float


def make_params(kernel_idx: int, hampel_idx: int, sigma_idx: int,
                kernel_choices: Sequence[int], window_choices: Sequence[int], sigma_choices: Sequence[float]) -> MedianHampelParams:
    kernel = kernel_choices[int(kernel_idx) % len(kernel_choices)]
    window = window_choices[int(hampel_idx) % len(window_choices)]
    sigma = sigma_choices[int(sigma_idx) % len(sigma_choices)]
    return MedianHampelParams(kernel_size=kernel, hampel_window=window, hampel_sigma=sigma)


def validate_params(params: MedianHampelParams) -> bool:
    return params.kernel_size >= 3 and params.kernel_size % 2 == 1 and params.hampel_window >= 3


def hampel_filter(signal: np.ndarray, window: int, n_sigmas: float) -> np.ndarray:
    n = len(signal)
    new = signal.copy()
    k = window // 2
    for i in range(k, n - k):
        segment = signal[i - k:i + k + 1]
        median = np.median(segment)
        mad = np.median(np.abs(segment - median)) + 1e-9
        threshold = n_sigmas * 1.4826 * mad
        if abs(signal[i] - median) > threshold:
            new[i] = median
    return new


def apply(signal: np.ndarray, params: MedianHampelParams) -> np.ndarray:
    filtered = hampel_filter(signal, params.hampel_window, params.hampel_sigma)
    return medfilt(filtered, kernel_size=params.kernel_size)


def apply_matrix(matrix: np.ndarray, params: MedianHampelParams) -> np.ndarray:
    return np.apply_along_axis(apply, 1, matrix, params)


def complexity(params: MedianHampelParams) -> float:
    return float(params.kernel_size + params.hampel_window)
