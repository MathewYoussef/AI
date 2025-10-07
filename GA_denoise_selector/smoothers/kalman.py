"""Simple scalar Kalman smoother treating wavelength as time."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class KalmanParams:
    process_var: float
    meas_var: float


def make_params(proc_idx: int, meas_idx: int, proc_choices: Sequence[float], meas_choices: Sequence[float]) -> KalmanParams:
    q = proc_choices[int(proc_idx) % len(proc_choices)]
    r = meas_choices[int(meas_idx) % len(meas_choices)]
    return KalmanParams(process_var=q, meas_var=r)


def validate_params(params: KalmanParams) -> bool:
    return params.process_var >= 0 and params.meas_var > 0


def _kalman(signal: np.ndarray, params: KalmanParams) -> np.ndarray:
    n = len(signal)
    x = 0.0
    P = 1.0
    Q = params.process_var
    R = params.meas_var
    estimates = np.zeros_like(signal)
    for i in range(n):
        # Prediction step
        P += Q
        # Update
        K = P / (P + R)
        x = x + K * (signal[i] - x)
        P = (1 - K) * P
        estimates[i] = x
    # RTS smoother pass
    smooth = estimates.copy()
    P = 1.0
    for i in range(n - 2, -1, -1):
        P = P + Q
        K = P / (P + R)
        smooth[i] = estimates[i] + K * (smooth[i + 1] - estimates[i])
    return smooth


def apply(signal: np.ndarray, params: KalmanParams) -> np.ndarray:
    return _kalman(signal, params)


def apply_matrix(matrix: np.ndarray, params: KalmanParams) -> np.ndarray:
    return np.vstack([apply(row, params) for row in matrix])


def complexity(params: KalmanParams) -> float:
    return float(params.process_var + params.meas_var)
