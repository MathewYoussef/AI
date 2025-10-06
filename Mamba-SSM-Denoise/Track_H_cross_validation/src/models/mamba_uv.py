from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict

import torch
from torch import nn

try:
    from mamba_ssm import Mamba
except ImportError as exc:  # pragma: no cover - requires optional dep
    raise ImportError(
        "mamba-ssm is not installed. Install it with `pip install \"mamba-ssm[causal-conv1d]\"`."
    ) from exc


@dataclass
class MambaUVConfig:
    sequence_length: int = 601
    in_channels: int = 3
    d_model: int = 128
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.0
    out_activation: str = "sigmoid"  # {sigmoid, clamp, none}
    geometry_film: bool = False
    film_hidden_dim: int = 32
    cond_dim: int = 1
    film_dropout: float = 0.0


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        if gamma is not None:
            x = x * gamma.unsqueeze(1)
        if beta is not None:
            x = x + beta.unsqueeze(1)
        x = self.mamba(x)
        x = self.dropout(x)
        return residual + x


LOGGER = logging.getLogger(__name__)


class MambaUV(nn.Module):
    """1-D SSU-Mamba variant for UV reflectance spectra."""

    def __init__(self, cfg: MambaUVConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.input_proj = nn.Conv1d(cfg.in_channels, cfg.d_model, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=cfg.d_model,
                    d_state=cfg.d_state,
                    d_conv=cfg.d_conv,
                    expand=cfg.expand,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.output_proj = nn.Conv1d(cfg.d_model, 1, kernel_size=1)

        self.geometry_film = bool(cfg.geometry_film)
        if self.geometry_film:
            hidden = int(cfg.film_hidden_dim)
            self.cond_dim = int(cfg.cond_dim)
            self.film_head = nn.Sequential(
                nn.Linear(self.cond_dim, hidden),
                nn.SiLU(),
                nn.Dropout(float(cfg.film_dropout)),
                nn.Linear(hidden, 2 * cfg.n_layers * cfg.d_model),
            )
            nn.init.zeros_(self.film_head[-1].weight)
            nn.init.zeros_(self.film_head[-1].bias)
            LOGGER.info(
                "MambaUV FiLM enabled (cond_dim=%d, hidden=%d, dropout=%.3f)",
                self.cond_dim,
                hidden,
                float(cfg.film_dropout),
            )
        else:
            self.cond_dim = 0
            LOGGER.info("MambaUV FiLM disabled; using identity gamma/beta")

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.input_proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.output_proj.bias)

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.out_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.cfg.out_activation == "clamp":
            return torch.clamp(x, 0.0, 1.0)
        return x

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        # Expect (B, C, L)
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B, C, L); got shape {tuple(x.shape)}")
        if x.size(-1) != self.cfg.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.cfg.sequence_length}; got {x.size(-1)}"
            )
        if x.size(1) != self.cfg.in_channels:
            raise ValueError(
                f"Expected {self.cfg.in_channels} input channel(s); got {x.size(1)}"
            )

        geometry_gamma = geometry_beta = None
        if self.geometry_film:
            if cond is not None:
                cond_input = cond
                if cond_input.dim() == 1:
                    cond_input = cond_input.unsqueeze(0)
                if cond_input.dim() != 2:
                    raise ValueError(
                        f"Conditioning tensor must have shape (B, cond_dim); got {tuple(cond_input.shape)}"
                    )
            else:
                angle_channel = x[:, -1]
                cos_theta = angle_channel.mean(dim=-1, keepdim=True)
                cond_input = cos_theta
            if cond_input.size(-1) != self.cond_dim:
                raise ValueError(
                    f"Expected conditioning dimension {self.cond_dim}; got {cond_input.size(-1)}"
                )
            cond_input = cond_input.to(x.dtype)
            film_params = self.film_head(cond_input)
            film_params = film_params.view(
                x.size(0), self.cfg.n_layers, 2, self.cfg.d_model
            )
            geometry_gamma = 1.0 + film_params[:, :, 0, :]
            geometry_beta = film_params[:, :, 1, :]

        x = self.input_proj(x)
        x = torch.relu(x)
        x = x.transpose(1, 2)  # -> (B, L, C)

        for idx, block in enumerate(self.blocks):
            gamma = beta = None
            if geometry_gamma is not None and geometry_beta is not None:
                gamma = geometry_gamma[:, idx, :]
                beta = geometry_beta[:, idx, :]
            x = block(x, gamma=gamma, beta=beta)

        x = self.final_norm(x)
        x = x.transpose(1, 2)  # -> (B, C, L)
        x = self.output_proj(x)
        return self._apply_activation(x)


MODEL_ZOO: Dict[str, MambaUVConfig] = {
    "mamba_tiny_uv": MambaUVConfig(d_model=128, n_layers=4, d_state=16, d_conv=4, expand=2),
    "mamba_small_uv": MambaUVConfig(d_model=192, n_layers=6, d_state=32, d_conv=4, expand=2),
}


def build_model(name: str, **overrides) -> MambaUV:
    if name not in MODEL_ZOO:
        raise KeyError(f"Unknown model '{name}'. Valid options: {list(MODEL_ZOO.keys())}")
    cfg_dict = MODEL_ZOO[name].__dict__.copy()
    cfg_dict.update(overrides)
    cfg = MambaUVConfig(**cfg_dict)
    return MambaUV(cfg)
