from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicalConsistencyLoss(nn.Module):
    def __init__(self, seq_len: int = 64):
        super().__init__()
        self.mapper = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
        )
        self.seq_len = seq_len

    def forward(self, p_v: torch.Tensor, p_r: torch.Tensor) -> torch.Tensor:
        # p_v, p_r: [B, T]
        dr = p_r[:, 1:] - p_r[:, :-1]
        dr = F.pad(dr, (0, 1), mode='replicate')
        mapped_v = self.mapper(p_v.unsqueeze(1)).squeeze(1)
        return F.l1_loss(dr, mapped_v)
