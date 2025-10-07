"""Cascade smoother utilities combining SG and wavelet."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .savitzky_golay import SGParams, apply_matrix as apply_sg_matrix, complexity as sg_complexity
from .wavelet import WaveletParams, apply_matrix as apply_wavelet_matrix, complexity as wavelet_complexity


@dataclass
class CascadeParams:
    sg_params: Optional[SGParams]
    wavelet_params: Optional[WaveletParams]
    order: str  # "sg->wav" or "wav->sg"


def make_params(sg_params: Optional[SGParams],
                wavelet_params: Optional[WaveletParams],
                order_flag: float) -> CascadeParams:
    order = "sg->wav" if order_flag < 0.5 else "wav->sg"
    return CascadeParams(sg_params=sg_params, wavelet_params=wavelet_params, order=order)


def apply_matrix(matrix: np.ndarray, params: CascadeParams) -> np.ndarray:
    result = matrix.copy()
    if params.order == "sg->wav":
        if params.sg_params is not None:
            result = apply_sg_matrix(result, params.sg_params)
        if params.wavelet_params is not None:
            result = apply_wavelet_matrix(result, params.wavelet_params)
    else:
        if params.wavelet_params is not None:
            result = apply_wavelet_matrix(result, params.wavelet_params)
        if params.sg_params is not None:
            result = apply_sg_matrix(result, params.sg_params)
    return result


def complexity(params: CascadeParams) -> float:
    total = 0.0
    if params.sg_params is not None:
        total += sg_complexity(params.sg_params)
    if params.wavelet_params is not None:
        total += wavelet_complexity(params.wavelet_params)
    return total
