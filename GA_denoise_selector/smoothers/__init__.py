"""Smoother registry for GA_denoise_selector."""

from .savitzky_golay import SGParams, make_params as make_sg_params, validate_params as validate_sg_params, apply_matrix as apply_sg, complexity as sg_complexity
from .wavelet import WaveletParams, AVAILABLE_FAMILIES, make_params as make_wavelet_params, validate_params as validate_wavelet_params, apply_matrix as apply_wavelet, complexity as wavelet_complexity
from .cascade import CascadeParams, make_params as make_cascade_params, apply_matrix as apply_cascade, complexity as cascade_complexity
from .butterworth import ButterworthParams, make_params as make_butter_params, validate_params as validate_butter_params, apply_matrix as apply_butterworth, complexity as butterworth_complexity
from .wiener_filter import WienerParams, make_params as make_wiener_params, validate_params as validate_wiener_params, apply_matrix as apply_wiener, complexity as wiener_complexity
from .median_hampel import MedianHampelParams, make_params as make_median_hampel_params, validate_params as validate_median_hampel_params, apply_matrix as apply_median_hampel, complexity as median_hampel_complexity
from .bilateral import BilateralParams, make_params as make_bilateral_params, validate_params as validate_bilateral_params, apply_matrix as apply_bilateral, complexity as bilateral_complexity
from .whittaker import WhittakerParams, make_params as make_whittaker_params, validate_params as validate_whittaker_params, apply_matrix as apply_whittaker, complexity as whittaker_complexity
from .smoothing_spline import SplineParams, make_params as make_spline_params, validate_params as validate_spline_params, apply_matrix as apply_spline, complexity as spline_complexity
from .total_variation import TVParams, make_params as make_tv_params, validate_params as validate_tv_params, apply_matrix as apply_tv, complexity as tv_complexity
from .gaussian_process import GPParams, make_params as make_gp_params, validate_params as validate_gp_params, apply_matrix as apply_gp, complexity as gp_complexity
from .kalman import KalmanParams, make_params as make_kalman_params, validate_params as validate_kalman_params, apply_matrix as apply_kalman, complexity as kalman_complexity
from .pca_smoother import PCAParams, make_params as make_pca_params, validate_params as validate_pca_params, apply_matrix as apply_pca, complexity as pca_complexity

__all__ = [
    "AVAILABLE_FAMILIES",
    # SG
    "SGParams", "make_sg_params", "validate_sg_params", "apply_sg", "sg_complexity",
    # Wavelet
    "WaveletParams", "make_wavelet_params", "validate_wavelet_params", "apply_wavelet", "wavelet_complexity",
    # Cascade
    "CascadeParams", "make_cascade_params", "apply_cascade", "cascade_complexity",
    # Butterworth
    "ButterworthParams", "make_butter_params", "validate_butter_params", "apply_butterworth", "butterworth_complexity",
    # Wiener
    "WienerParams", "make_wiener_params", "validate_wiener_params", "apply_wiener", "wiener_complexity",
    # Median-Hampel
    "MedianHampelParams", "make_median_hampel_params", "validate_median_hampel_params", "apply_median_hampel", "median_hampel_complexity",
    # Bilateral
    "BilateralParams", "make_bilateral_params", "validate_bilateral_params", "apply_bilateral", "bilateral_complexity",
    # Whittaker
    "WhittakerParams", "make_whittaker_params", "validate_whittaker_params", "apply_whittaker", "whittaker_complexity",
    # Spline
    "SplineParams", "make_spline_params", "validate_spline_params", "apply_spline", "spline_complexity",
    # Total variation
    "TVParams", "make_tv_params", "validate_tv_params", "apply_tv", "tv_complexity",
    # Gaussian Process
    "GPParams", "make_gp_params", "validate_gp_params", "apply_gp", "gp_complexity",
    # Kalman
    "KalmanParams", "make_kalman_params", "validate_kalman_params", "apply_kalman", "kalman_complexity",
    # PCA
    "PCAParams", "make_pca_params", "validate_pca_params", "apply_pca", "pca_complexity",
]
