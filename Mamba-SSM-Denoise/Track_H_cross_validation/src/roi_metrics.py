"""ROI-focused spectral metrics and dip-shape analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # optional SciPy import for automatic dip detection
    from scipy.signal import find_peaks
except ImportError:  # pragma: no cover - optional dependency
    find_peaks = None

__all__ = [
    "DipSummary",
    "aggregate_roi_metrics",
    "compute_dip_metrics",
    "psnr_roi",
    "sam_deg",
]


def _mask_roi(lam_nm: np.ndarray, roi: Tuple[float, float]) -> np.ndarray:
    return (lam_nm >= roi[0]) & (lam_nm <= roi[1])


def sam_deg(pred: np.ndarray, targ: np.ndarray, mask: np.ndarray) -> float:
    x = pred[..., mask].astype(np.float64)
    y = targ[..., mask].astype(np.float64)
    dot = (x * y).sum(-1)
    nx = np.sqrt((x**2).sum(-1) + 1e-12)
    ny = np.sqrt((y**2).sum(-1) + 1e-12)
    cosine = np.clip(dot / (nx * ny + 1e-12), -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    return float(np.mean(angles))


def psnr_roi(pred: np.ndarray, targ: np.ndarray, mask: np.ndarray) -> float:
    x = pred[..., mask].astype(np.float64)
    y = targ[..., mask].astype(np.float64)
    mse = np.mean((x - y) ** 2)
    return float(-10.0 * np.log10(mse + 1e-12))


@dataclass
class DipSummary:
    n_dips: int
    depth_mae: float
    centroid_mae_nm: float
    area_mape_pct: float


def _fit_polynomial_baseline(
    lam_window: np.ndarray,
    spectrum_window: np.ndarray,
    center_lambda: float,
    guard_nm: float,
    poly_order: int,
) -> np.ndarray:
    """Fit a polynomial baseline around a dip, ignoring the guarded core."""

    lam_window = lam_window.astype(np.float64)
    spectrum_window = spectrum_window.astype(np.float64)

    if guard_nm > 0.0:
        mask = np.abs(lam_window - center_lambda) >= guard_nm
    else:
        mask = np.ones_like(lam_window, dtype=bool)

    if mask.sum() < poly_order + 1:
        mask = np.ones_like(lam_window, dtype=bool)

    order = int(np.clip(poly_order, 0, mask.sum() - 1))
    if order < 1:
        baseline_level = float(np.median(spectrum_window[mask]))
        return np.full_like(spectrum_window, baseline_level, dtype=np.float64)

    try:
        coeffs = np.polyfit(lam_window[mask], spectrum_window[mask], deg=order)
        baseline = np.polyval(coeffs, lam_window)
    except np.linalg.LinAlgError:
        baseline = np.full_like(spectrum_window, float(np.median(spectrum_window[mask])), dtype=np.float64)

    return baseline



def _continuum_remove(
    spectrum: np.ndarray,
    lam_nm: np.ndarray,
    start: int,
    end: int,
    guard_idx: int,
    eps: float = 1e-6,
) -> np.ndarray:
    seg = spectrum[start : end + 1]
    lam_segment = lam_nm[start : end + 1]

    left = max(start, start - guard_idx)
    right = min(end, end + guard_idx)

    y0 = spectrum[left]
    y1 = spectrum[right]
    lam0 = lam_nm[left]
    lam1 = lam_nm[right]

    t = (lam_segment - lam0) / (lam1 - lam0 + eps)
    continuum = (1.0 - t) * y0 + t * y1
    continuum = np.clip(continuum, eps, None)
    cr = np.clip(seg / continuum, 0.0, 2.0)
    depth = np.clip(1.0 - cr, 0.0, 1.0)
    return depth


def _centroid(lam_nm: np.ndarray, absorption: np.ndarray) -> float:
    weights = np.clip(absorption.astype(np.float64), 0.0, None)
    total = weights.sum() + 1e-12
    return float((lam_nm * weights).sum() / total)


def _parabolic_centroid(lam_nm: np.ndarray, absorption: np.ndarray) -> float:
    if absorption.size < 3:
        return _centroid(lam_nm, absorption)
    idx = int(np.argmax(absorption))
    if idx == 0 or idx == absorption.size - 1:
        return float(lam_nm[idx])
    y0 = float(absorption[idx])
    y_minus = float(absorption[idx - 1])
    y_plus = float(absorption[idx + 1])
    denom = (y_minus - 2.0 * y0 + y_plus)
    if abs(denom) < 1e-12:
        return float(lam_nm[idx])
    delta = 0.5 * (y_minus - y_plus) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    sample_idx = np.arange(lam_nm.size, dtype=np.float64)
    return float(np.interp(idx + delta, sample_idx, lam_nm.astype(np.float64)))


def _area(absorption: np.ndarray, lam_nm: np.ndarray) -> float:
    return float(np.trapz(np.clip(absorption.astype(np.float64), 0.0, None), lam_nm))


def _detect_centers(
    target: np.ndarray,
    lam_nm: np.ndarray,
    roi: Tuple[float, float],
    max_dips: int,
    min_prominence: float,
) -> List[int]:
    roi_mask = _mask_roi(lam_nm, roi)
    roi_indices = np.where(roi_mask)[0]
    if roi_indices.size == 0:
        return []
    if find_peaks is None:
        raise RuntimeError(
            "scipy is required for automatic dip detection; install scipy or provide known_lines_nm."
        )
    inv = -target[roi_indices]
    peaks, props = find_peaks(inv, prominence=min_prominence)
    if peaks.size == 0:
        return []
    if max_dips > 0 and peaks.size > max_dips:
        prominences = props.get("prominences")
        if prominences is not None:
            order = np.argsort(prominences)[-max_dips:]
            peaks = np.sort(peaks[order])
        else:
            peaks = peaks[:max_dips]
    return roi_indices[peaks].tolist()


def _enforce_min_spacing(centers: List[int], lam_nm: np.ndarray, min_separation_nm: float) -> List[int]:
    if min_separation_nm <= 0.0 or len(centers) <= 1:
        return sorted(centers)

    filtered: List[int] = []
    for idx in sorted(centers, key=lambda i: lam_nm[i]):
        if filtered and float(lam_nm[idx] - lam_nm[filtered[-1]]) < min_separation_nm - 1e-9:
            continue
        filtered.append(idx)
    return filtered


def compute_dip_metrics(
    pred: np.ndarray,
    targ: np.ndarray,
    lam_nm: np.ndarray,
    roi: Tuple[float, float] = (320.0, 500.0),
    known_lines_nm: Optional[List[float]] = None,
    half_width_nm: float = 2.0,
    min_prominence: float = 0.002,
    baseline: str = "local",
    baseline_guard_nm: float = 5.0,
    centroid_method: str = "grid",
    max_dips: int = 6,
    poly_order: int = 2,
    min_separation_nm: float = 0.0,
) -> Tuple[DipSummary, List[Dict[str, float]]]:
    baseline = baseline.lower()
    if baseline not in {"local", "continuum", "flat"}:
        raise ValueError(f"Unsupported baseline '{baseline}'")

    step_nm = float(lam_nm[1] - lam_nm[0]) if lam_nm.size > 1 else 1.0
    half_width_idx = max(1, int(round(half_width_nm / max(step_nm, 1e-6))))
    guard_idx = max(0, int(round(baseline_guard_nm / max(step_nm, 1e-6))))
    centroid_method = centroid_method.lower()

    if known_lines_nm:
        centers = [int(np.abs(lam_nm - c).argmin()) for c in known_lines_nm if roi[0] <= c <= roi[1]]
    else:
        centers = _detect_centers(targ, lam_nm, roi, max_dips, min_prominence)

    centers = _enforce_min_spacing(centers, lam_nm, min_separation_nm)
    centers = sorted(centers, key=lambda i: lam_nm[i])
    if max_dips > 0:
        centers = centers[:max_dips]

    if not centers:
        return DipSummary(0, 0.0, 0.0, 0.0), []

    area_errs: List[float] = []
    centroid_errs: List[float] = []
    depth_errs: List[float] = []
    dip_records: List[Dict[str, float]] = []

    for local_id, center_idx in enumerate(centers, start=1):
        if not (roi[0] <= lam_nm[center_idx] <= roi[1]):
            continue
        start = max(0, center_idx - half_width_idx)
        end = min(pred.shape[-1] - 1, center_idx + half_width_idx)
        if end <= start:
            continue

        window_idx = np.arange(start, end + 1)
        lam_window = lam_nm[window_idx]
        pred_window = pred[window_idx]
        targ_window = targ[window_idx]

        if baseline == "continuum":
            absorption_t = _continuum_remove(targ, lam_nm, start, end, guard_idx)
            absorption_p = _continuum_remove(pred, lam_nm, start, end, guard_idx)
        elif baseline == "flat":
            absorption_t = np.clip(1.0 - targ_window, 0.0, 1.0)
            absorption_p = np.clip(1.0 - pred_window, 0.0, 1.0)
        else:
            baseline_t = _fit_polynomial_baseline(
                lam_window,
                targ_window,
                float(lam_nm[center_idx]),
                baseline_guard_nm,
                poly_order,
            )
            baseline_p = _fit_polynomial_baseline(
                lam_window,
                pred_window,
                float(lam_nm[center_idx]),
                baseline_guard_nm,
                poly_order,
            )
            absorption_t = np.clip(baseline_t - targ_window, 0.0, None)
            absorption_p = np.clip(baseline_p - pred_window, 0.0, None)

        area_t = _area(absorption_t, lam_window)
        if area_t < 1e-12:
            continue
        area_p = _area(absorption_p, lam_window)
        area_err = 100.0 * abs(area_p - area_t) / (area_t + 1e-12)
        area_errs.append(area_err)

        if centroid_method == "parabolic":
            centroid_t = _parabolic_centroid(lam_window, absorption_t)
            centroid_p = _parabolic_centroid(lam_window, absorption_p)
        else:
            centroid_t = _centroid(lam_window, absorption_t)
            centroid_p = _centroid(lam_window, absorption_p)
        centroid_err = abs(centroid_p - centroid_t)
        centroid_errs.append(centroid_err)

        depth_mae = float(np.mean(np.abs(absorption_p - absorption_t)))
        depth_errs.append(depth_mae)

        dip_records.append({
            "dip_id": local_id,
            "center_nm": float(lam_nm[center_idx]),
            "center_index": int(center_idx),
            "window_start_index": int(start),
            "window_end_index": int(end),
            "target_centroid_nm": float(centroid_t),
            "pred_centroid_nm": float(centroid_p),
            "centroid_error_nm": float(centroid_err),
            "target_area": float(area_t),
            "pred_area": float(area_p),
            "area_error_pct": float(area_err),
            "depth_mae": depth_mae,
        })

    if not area_errs:
        return DipSummary(0, 0.0, 0.0, 0.0), []

    summary = DipSummary(
        n_dips=len(area_errs),
        depth_mae=float(np.median(depth_errs)),
        centroid_mae_nm=float(np.median(centroid_errs)),
        area_mape_pct=float(np.median(area_errs)),
    )
    return summary, dip_records


def aggregate_roi_metrics(records: List[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {}
    keys = [key for key, value in records[0].items() if isinstance(value, (int, float))]
    return {key: float(np.median([rec[key] for rec in records])) for key in keys}
