"""Whittaker-Eilers smoothing."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass
class WhittakerParams:
    lam: float
    diff_order: int


def make_params(lam_idx: int, diff_idx: int, lam_choices: Sequence[float], diff_choices: Sequence[int]) -> WhittakerParams:
    lam = lam_choices[int(lam_idx) % len(lam_choices)]
    diff = diff_choices[int(diff_idx) % len(diff_choices)]
    return WhittakerParams(lam=lam, diff_order=diff)


def validate_params(params: WhittakerParams) -> bool:
    return params.lam > 0 and params.diff_order >= 1


def _difference_matrix(m: int, order: int) -> sparse.csc_matrix:
    if order == 0:
        return sparse.eye(m, format="csc")
    diff = np.diff(np.eye(m), n=order, axis=0)
    return sparse.csc_matrix(diff)


def _whittaker_smooth(y: np.ndarray, lam: float, d: int) -> np.ndarray:
    m = y.shape[0]
    E = sparse.eye(m, format="csc")
    D = _difference_matrix(m, d)
    penalty = lam * (D.T @ D)
    return spsolve(E + penalty, y)


def apply(signal: np.ndarray, params: WhittakerParams) -> np.ndarray:
    return _whittaker_smooth(signal, params.lam, params.diff_order)


def apply_matrix(matrix: np.ndarray, params: WhittakerParams) -> np.ndarray:
    return np.vstack([apply(row, params) for row in matrix])


def complexity(params: WhittakerParams) -> float:
    return float(np.log10(params.lam + 1) + params.diff_order)
