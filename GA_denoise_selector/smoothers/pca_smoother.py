"""PCA reconstruction smoother."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class PCAParams:
    n_components: int


def make_params(idx: int, choices: Sequence[int]) -> PCAParams:
    return PCAParams(n_components=choices[int(idx) % len(choices)])


def validate_params(params: PCAParams, n_features: int) -> bool:
    return 1 <= params.n_components <= n_features


def apply_matrix(matrix: np.ndarray, params: PCAParams) -> np.ndarray:
    pca = PCA(n_components=params.n_components)
    transformed = pca.fit_transform(matrix)
    reconstructed = pca.inverse_transform(transformed)
    return reconstructed


def complexity(params: PCAParams) -> float:
    return float(params.n_components)
