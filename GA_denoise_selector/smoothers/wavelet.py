"""Wavelet denoiser utilities."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pywt

AVAILABLE_FAMILIES: Sequence[str] = ("sym4", "sym6", "sym8", "coif3", "db4", "db5")


@dataclass
class WaveletParams:
    family: str
    level: int


def make_params(family: str, level: int) -> WaveletParams:
    return WaveletParams(family=str(family), level=int(round(level)))


def validate_params(params: WaveletParams, signal_length: int) -> bool:
    if params.level < 1:
        return False
    try:
        wavelet = pywt.Wavelet(params.family)
    except ValueError:
        return False
    max_level = pywt.dwt_max_level(signal_length, wavelet.dec_len)
    return params.level <= max_level


def apply(signal: np.ndarray, params: WaveletParams) -> np.ndarray:
    coeffs = pywt.wavedec(signal, params.family, mode="periodization", level=params.level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs[1:] = [pywt.threshold(c, value=threshold, mode="hard") for c in coeffs[1:]]
    return pywt.waverec(coeffs, params.family, mode="periodization")[: len(signal)]


def apply_matrix(matrix: np.ndarray, params: WaveletParams) -> np.ndarray:
    return np.apply_along_axis(apply, 1, matrix, params)


def complexity(params: WaveletParams) -> float:
    return float(params.level * 10)
