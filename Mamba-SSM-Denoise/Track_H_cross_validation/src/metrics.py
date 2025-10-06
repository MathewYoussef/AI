import math
from typing import Dict

import torch
import torch.nn.functional as F


def psnr(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(prediction, target, reduction="mean")
    mse = torch.clamp_min(mse, torch.finfo(mse.dtype).tiny)
    max_val = torch.tensor(data_range, dtype=prediction.dtype, device=prediction.device)
    return 20.0 * torch.log10(max_val) - 10.0 * torch.log10(mse)


def spectral_angle_mapper(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Flatten channel/sequence dims to vectors per example
    pred_vec = prediction.flatten(start_dim=1)
    tgt_vec = target.flatten(start_dim=1)

    dot = (pred_vec * tgt_vec).sum(dim=1)
    pred_norm = torch.norm(pred_vec, dim=1)
    tgt_norm = torch.norm(tgt_vec, dim=1)

    cosine = dot / (pred_norm * tgt_norm + eps)
    cosine = cosine.clamp(-1.0 + eps, 1.0 - eps)
    angles = torch.arccos(cosine)
    return angles.mean() * (180.0 / math.pi)


def compute_metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "psnr": psnr(prediction, target),
        "sam_deg": spectral_angle_mapper(prediction, target),
    }
