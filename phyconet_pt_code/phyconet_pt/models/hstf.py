from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMapTokenModulation(nn.Module):
    """Lightweight token-to-feature modulation.

    This avoids dense attention over all spatial positions and makes ablations easier.
    """
    def __init__(self, channels: int, token_dim: int):
        super().__init__()
        self.to_scale = nn.Sequential(
            nn.Linear(token_dim, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.to_bias = nn.Sequential(
            nn.Linear(token_dim, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.out_norm = nn.BatchNorm2d(channels)
        self.out_act = nn.GELU()

    def forward(self, fmap: torch.Tensor, shared_tokens: torch.Tensor) -> torch.Tensor:
        context = shared_tokens.mean(dim=1)  # [B, Dt]
        scale = self.to_scale(context).unsqueeze(-1).unsqueeze(-1)
        bias = self.to_bias(context).unsqueeze(-1).unsqueeze(-1)
        out = fmap * (1.0 + scale) + bias
        return self.out_act(self.out_norm(out))


class HSTFBlock(nn.Module):
    def __init__(self, channels: int, token_dim: int = 128, token_grid: Tuple[int, int] = (4, 4)):
        super().__init__()
        self.grid = token_grid
        self.d_proj = nn.Linear(channels, token_dim)
        self.c_proj = nn.Linear(channels, token_dim)
        self.gate_d = nn.Sequential(nn.Linear(token_dim, token_dim), nn.Sigmoid())
        self.gate_c = nn.Sequential(nn.Linear(token_dim, token_dim), nn.Sigmoid())
        self.d_back = FeatureMapTokenModulation(channels, token_dim)
        self.c_back = FeatureMapTokenModulation(channels, token_dim)
        self.shared_norm = nn.LayerNorm(token_dim)

    def _to_tokens(self, fmap: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(fmap, self.grid)
        tokens = pooled.flatten(2).transpose(1, 2)
        return proj(tokens)

    def forward(self, fd: torch.Tensor, fc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        td = self._to_tokens(fd, self.d_proj)
        tc = self._to_tokens(fc, self.c_proj)
        gd = self.gate_d(td.mean(dim=1, keepdim=True))
        gc = self.gate_c(tc.mean(dim=1, keepdim=True))
        ts = self.shared_norm(gd * td + gc * tc)
        fd2 = self.d_back(fd, ts)
        fc2 = self.c_back(fc, ts)
        sd = td.mean(dim=1)
        sc = tc.mean(dim=1)
        ss = ts.mean(dim=1)
        return fd2, fc2, sd, sc, ss
