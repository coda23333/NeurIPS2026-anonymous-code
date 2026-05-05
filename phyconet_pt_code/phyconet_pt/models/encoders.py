from __future__ import annotations

from typing import List
import torch
import torch.nn as nn

from .blocks import ConvBNAct, ResBlock2D, TemporalMHSA2D, LocalWindowAttention2D


class DFSEncoder(nn.Module):
    def __init__(self, use_temporal_mhsa: bool = True):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(1, 32, 3, stride=1, padding=1),
            ConvBNAct(32, 64, 3, stride=(2, 1), padding=1),
        )
        self.stage1 = nn.Sequential(ResBlock2D(64, 64), ResBlock2D(64, 64))
        self.stage2 = nn.Sequential(ResBlock2D(64, 128, stride=(2, 2)), ResBlock2D(128, 128))
        self.stage3 = nn.Sequential(ResBlock2D(128, 256, stride=(2, 2)), ResBlock2D(256, 256))
        self.stage4 = nn.Sequential(ResBlock2D(256, 256, stride=(2, 2)), ResBlock2D(256, 256))
        self.use_temporal_mhsa = use_temporal_mhsa
        self.temporal3 = TemporalMHSA2D(256) if use_temporal_mhsa else nn.Identity()
        self.temporal4 = TemporalMHSA2D(256) if use_temporal_mhsa else nn.Identity()
        self.out_dim = 256

    def forward_stages(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.temporal3(self.stage3(s2))
        s4 = self.temporal4(self.stage4(s3))
        return [s1, s2, s3, s4]


class CIREncoder(nn.Module):
    def __init__(self, use_local_attn: bool = True):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(1, 32, 5, stride=(2, 2), padding=2),
            ConvBNAct(32, 64, 3, stride=(2, 1), padding=1),
        )
        self.stage1 = nn.Sequential(ResBlock2D(64, 64), ResBlock2D(64, 64))
        self.stage2 = nn.Sequential(ResBlock2D(64, 128, stride=(2, 2)), ResBlock2D(128, 128))
        self.stage3 = nn.Sequential(ResBlock2D(128, 256, stride=(2, 2)), ResBlock2D(256, 256))
        self.stage4 = nn.Sequential(ResBlock2D(256, 256, stride=(2, 2)), ResBlock2D(256, 256))
        self.use_local_attn = use_local_attn
        self.local4 = LocalWindowAttention2D(256) if use_local_attn else nn.Identity()
        self.out_dim = 256

    def forward_stages(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.local4(self.stage4(s3))
        return [s1, s2, s3, s4]
