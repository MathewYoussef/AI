from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


class CharbonnierLoss(torch.nn.Module):
    """Robust L1 loss often used for denoising."""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diff = prediction - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if weight is not None:
            loss = loss * weight.view(1, 1, -1)
        return loss.mean()


def spectral_derivative(x: torch.Tensor, order: int = 1, delta: float = 1.0) -> torch.Tensor:
    """Finite-difference derivative along the spectral axis."""

    if order < 1:
        raise ValueError("order must be >= 1")

    deriv = x
    for _ in range(order):
        deriv = (deriv[..., 1:] - deriv[..., :-1]) / delta
    return deriv


@dataclass
class DerivativeLossConfig:
    order: int = 1
    weight: float = 0.3
    reduction: str = "mean"
    clamp_min: Optional[float] = None


class DerivativeLoss(torch.nn.Module):
    """Matches derivatives between prediction and target spectra."""

    def __init__(self, cfg: DerivativeLossConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        delta_lambda: float = 1.0,
    ) -> torch.Tensor:
        pred_d = spectral_derivative(prediction, order=self.cfg.order, delta=delta_lambda)
        target_d = spectral_derivative(target, order=self.cfg.order, delta=delta_lambda)

        diff = torch.abs(pred_d - target_d)
        if weight is not None:
            diff = diff * weight.view(1, 1, -1)

        if self.cfg.reduction == "mean":
            loss = diff.mean()
        elif self.cfg.reduction == "sum":
            loss = diff.sum()
        else:
            raise ValueError(f"Unsupported reduction '{self.cfg.reduction}'")

        if self.cfg.clamp_min is not None:
            loss = torch.clamp_min(loss, self.cfg.clamp_min)
        return loss * self.cfg.weight


class CombinedLoss(torch.nn.Module):
    """Weighted Charbonnier + derivative-aware loss."""

    def __init__(
        self,
        charbonnier_eps: float = 1e-3,
        derivative_cfg: Optional[DerivativeLossConfig] = None,
        lambda_weights: Optional[torch.Tensor] = None,
        delta_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        self.charbonnier = CharbonnierLoss(eps=charbonnier_eps)
        self.derivative = DerivativeLoss(derivative_cfg or DerivativeLossConfig())
        self.delta_lambda = float(delta_lambda)

        if lambda_weights is not None:
            weight_tensor = torch.as_tensor(lambda_weights, dtype=torch.float32)
        else:
            weight_tensor = torch.tensor([], dtype=torch.float32)
        self.register_buffer("charb_weights", weight_tensor)

        if weight_tensor.numel() >= 2:
            mid = 0.5 * (weight_tensor[1:] + weight_tensor[:-1])
        else:
            mid = torch.tensor([], dtype=torch.float32)
        self.register_buffer("derivative_weights", mid)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        charb_weight = None if self.charb_weights.numel() == 0 else self.charb_weights.to(prediction.device)
        deriv_weight = None if self.derivative_weights.numel() == 0 else self.derivative_weights.to(prediction.device)

        base = self.charbonnier(prediction, target, weight=charb_weight)
        deriv = self.derivative(
            prediction,
            target,
            weight=deriv_weight,
            delta_lambda=self.delta_lambda,
        )
        return base + deriv


@dataclass
class DipLossConfig:
    roi_nm: Tuple[float, float] = (320.0, 500.0)
    m_dips: int = 6
    window_half_nm: float = 5.0
    min_area: float = 1e-5
    w_area: float = 1.0
    w_equivalent_width: float = 0.0
    w_centroid: float = 1.0
    w_depth: float = 0.2
    underfill_factor: float = 2.0
    detect_sigma_nm: float = 1.0
    lambda_step_nm: float = 0.5
    baseline: str = "continuum"
    baseline_guard_nm: float = 5.0
    w_curvature: float = 0.0


def _log_kernel(sigma_samples: float, device: torch.device) -> torch.Tensor:
    radius = int(max(1.0, 3.0 * sigma_samples))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    sigma_sq = sigma_samples ** 2
    gaussian = torch.exp(-x**2 / (2.0 * sigma_sq))
    log = (x**2 - sigma_sq) / (sigma_sq**2) * gaussian
    log = log - log.mean()
    return log.view(1, 1, -1)


def _continuum_remove_torch(
    spectra: torch.Tensor,
    lam_nm: torch.Tensor,
    idx_start: int,
    idx_end: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    segment = spectra[..., idx_start : idx_end + 1]
    lam_segment = lam_nm[idx_start : idx_end + 1]

    y0 = spectra[..., idx_start].unsqueeze(-1)
    y1 = spectra[..., idx_end].unsqueeze(-1)
    lam0 = lam_nm[idx_start]
    lam1 = lam_nm[idx_end]

    t = (lam_segment - lam0) / (lam1 - lam0 + eps)
    continuum = (1.0 - t) * y0 + t * y1
    continuum = torch.clamp(continuum, min=eps)
    cr = torch.clamp(segment / continuum, 0.0, 2.0)
    depth = torch.clamp(1.0 - cr, 0.0, 1.0)
    return depth


def _local_absorption_torch(
    spectra: torch.Tensor,
    idx_start: int,
    idx_end: int,
    guard_count: int,
) -> torch.Tensor:
    segment = spectra[..., idx_start : idx_end + 1]
    length = segment.shape[-1]
    if guard_count <= 0 or guard_count * 2 >= length:
        left = segment[..., :1]
        right = segment[..., -1:]
    else:
        left = segment[..., :guard_count].mean(dim=-1, keepdim=True)
        right = segment[..., -guard_count:].mean(dim=-1, keepdim=True)

    weights = torch.linspace(
        0.0, 1.0, steps=length, device=segment.device, dtype=segment.dtype
    ).view(*([1] * (segment.dim() - 1)), -1)
    baseline = torch.clamp(left + (right - left) * weights, min=1e-6)
    return torch.clamp(baseline - segment, min=0.0)


class DipAwareLoss(torch.nn.Module):
    """Differentiable dip-shape preservation loss."""

    def __init__(self, lam_nm: torch.Tensor, cfg: DipLossConfig) -> None:
        super().__init__()
        self.register_buffer("lam_nm", lam_nm.float(), persistent=False)
        cfg.baseline = cfg.baseline.lower()
        self.cfg = cfg

        sigma_samples = max(1.0, cfg.detect_sigma_nm / max(cfg.lambda_step_nm, 1e-6))
        kernel = _log_kernel(sigma_samples, lam_nm.device)
        self.register_buffer("log_kernel", kernel, persistent=False)

        mask = (lam_nm >= cfg.roi_nm[0]) & (lam_nm <= cfg.roi_nm[1])
        self.register_buffer("roi_mask", mask.float(), persistent=False)

    @torch.no_grad()
    def _detect_centers(self, target: torch.Tensor) -> List[torch.Tensor]:
        scores = -F.conv1d(
            target.unsqueeze(1), self.log_kernel, padding=self.log_kernel.shape[-1] // 2
        ).squeeze(1)
        scores = scores * self.roi_mask

        pooled = F.max_pool1d(scores.unsqueeze(1), kernel_size=11, stride=1, padding=5).squeeze(1)
        keep_mask = (scores == pooled) & (scores > scores.mean(dim=1, keepdim=True))

        centers: List[torch.Tensor] = []
        for b in range(scores.shape[0]):
            candidates = torch.where(keep_mask[b])[0]
            if candidates.numel() == 0:
                centers.append(torch.empty(0, dtype=torch.long, device=scores.device))
                continue
            topk = torch.topk(
                scores[b, candidates], k=min(self.cfg.m_dips, candidates.numel())
            ).indices
            centers.append(candidates[topk])
        return centers

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.dim() != 2 or target.dim() != 2:
            raise ValueError("DipAwareLoss expects inputs shaped (B, L)")

        pred = prediction.float()
        tgt = target.float()
        lam_nm = self.lam_nm
        cfg = self.cfg
        step = max(cfg.lambda_step_nm, 1e-6)
        half_window = max(1, int(round(cfg.window_half_nm / step)))
        guard_count = max(1, int(round(cfg.baseline_guard_nm / step)))

        centers = self._detect_centers(tgt)

        area_terms: List[torch.Tensor] = []
        eq_width_terms: List[torch.Tensor] = []
        centroid_terms: List[torch.Tensor] = []
        depth_terms: List[torch.Tensor] = []
        curvature_terms: List[torch.Tensor] = []

        for batch_idx, center_idxs in enumerate(centers):
            for center in center_idxs:
                start = int(max(0, center.item() - half_window))
                end = int(min(pred.shape[1] - 1, center.item() + half_window))
                if end <= start:
                    continue

                if cfg.baseline == "continuum":
                    depth_t = _continuum_remove_torch(
                        tgt[batch_idx : batch_idx + 1], lam_nm, start, end
                    )[0]
                    depth_p = _continuum_remove_torch(
                        pred[batch_idx : batch_idx + 1], lam_nm, start, end
                    )[0]
                elif cfg.baseline == "flat":
                    segment_t = tgt[batch_idx, start : end + 1]
                    segment_p = pred[batch_idx, start : end + 1]
                    depth_t = torch.clamp(1.0 - segment_t, min=0.0)
                    depth_p = torch.clamp(1.0 - segment_p, min=0.0)
                else:  # local baseline
                    window_guard = min(guard_count, (end - start + 1) // 2)
                    window_guard = max(window_guard, 1)
                    depth_t = _local_absorption_torch(
                        tgt[batch_idx : batch_idx + 1], start, end, window_guard
                    )[0]
                    depth_p = _local_absorption_torch(
                        pred[batch_idx : batch_idx + 1], start, end, window_guard
                    )[0]

                area_t = torch.trapz(depth_t, lam_nm[start : end + 1])
                if area_t < cfg.min_area:
                    continue

                area_p = torch.trapz(depth_p, lam_nm[start : end + 1])
                rel_err = torch.abs(area_p - area_t) / (area_t + 1e-7)
                underfill = torch.clamp(area_t - area_p, min=0.0) / (area_t + 1e-7)
                area_terms.append(rel_err + (cfg.underfill_factor - 1.0) * underfill)

                if cfg.w_equivalent_width > 0.0:
                    window_nm = (end - start + 1) * step
                    eq_t = area_t / (window_nm + 1e-7)
                    eq_p = area_p / (window_nm + 1e-7)
                    eq_err = torch.abs(eq_p - eq_t) / (eq_t + 1e-7)
                    eq_underfill = torch.clamp(eq_t - eq_p, min=0.0) / (eq_t + 1e-7)
                    eq_width_terms.append(
                        eq_err + (cfg.underfill_factor - 1.0) * eq_underfill
                    )

                weights_t = depth_t + 1e-7
                centroid_t = (lam_nm[start : end + 1] * weights_t).sum() / weights_t.sum()
                weights_p = depth_p + 1e-7
                centroid_p = (lam_nm[start : end + 1] * weights_p).sum() / weights_p.sum()
                centroid_terms.append(torch.abs(centroid_p - centroid_t))

                depth_terms.append(torch.mean(torch.abs(depth_p - depth_t)))

                if cfg.w_curvature > 0.0 and depth_t.numel() >= 3:
                    curvature_t = torch.abs(
                        depth_t[2:] - 2.0 * depth_t[1:-1] + depth_t[:-2]
                    ) / (step * step)
                    curvature_p = torch.abs(
                        depth_p[2:] - 2.0 * depth_p[1:-1] + depth_p[:-2]
                    ) / (step * step)
                    curvature_terms.append(
                        torch.clamp(curvature_t - curvature_p, min=0.0).mean()
                    )

        if not area_terms:
            return torch.zeros((), dtype=pred.dtype, device=pred.device)

        area_loss = torch.stack(area_terms).mean()
        if eq_width_terms:
            eq_loss = torch.stack(eq_width_terms).mean()
        else:
            eq_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
        centroid_loss = torch.stack(centroid_terms).mean()
        depth_loss = torch.stack(depth_terms).mean()
        curvature_loss = (
            torch.stack(curvature_terms).mean()
            if curvature_terms
            else torch.zeros((), dtype=pred.dtype, device=pred.device)
        )

        return (
            cfg.w_area * area_loss
            + cfg.w_equivalent_width * eq_loss
            + cfg.w_centroid * centroid_loss
            + cfg.w_depth * depth_loss
            + cfg.w_curvature * curvature_loss
        )
