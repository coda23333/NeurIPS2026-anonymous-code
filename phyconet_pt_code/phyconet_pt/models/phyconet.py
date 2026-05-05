from __future__ import annotations

from typing import Any, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import DFSEncoder, CIREncoder
from .hstf import HSTFBlock
from .blocks import ProjectionHead, MLP


class PhyCoNetPT(nn.Module):
    def __init__(self, model_cfg: Dict[str, Any]):
        super().__init__()
        token_dim = int(model_cfg.get('token_dim', 128))
        token_grid = tuple(model_cfg.get('token_grid', [4, 4]))
        proj_dim = int(model_cfg.get('proj_dim', 256))

        self.dfs_encoder = DFSEncoder(use_temporal_mhsa=bool(model_cfg.get('use_temporal_mhsa_dfs', True)))
        self.cir_encoder = CIREncoder(use_local_attn=bool(model_cfg.get('use_local_attn_cir', True)))

        self.use_hstf = bool(model_cfg.get('use_hstf', True))
        self.hstf_stage_flags = list(model_cfg.get('hstf_stage_flags', [True, True, True, True]))
        if len(self.hstf_stage_flags) != 4:
            raise ValueError('hstf_stage_flags must have length 4')

        self.hstf_blocks = nn.ModuleList([
            HSTFBlock(channels=64, token_dim=token_dim, token_grid=token_grid),
            HSTFBlock(channels=128, token_dim=token_dim, token_grid=token_grid),
            HSTFBlock(channels=256, token_dim=token_dim, token_grid=token_grid),
            HSTFBlock(channels=256, token_dim=token_dim, token_grid=token_grid),
        ])

        self.d_proj = ProjectionHead(256, proj_dim)
        self.c_proj = ProjectionHead(256, proj_dim)

        self.motion_head = nn.Conv1d(256, 1, kernel_size=1)
        self.range_head = nn.Conv1d(256, 1, kernel_size=1)

        self.out_dim = 256 + 256 + token_dim
        self.proj_dim = proj_dim
        self.token_dim = token_dim

    @staticmethod
    def _gap(x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def _apply_hstf(
        self,
        dfs_stages: List[torch.Tensor],
        cir_stages: List[torch.Tensor],
    ) -> tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        new_d, new_c = [], []
        stage_d_tokens, stage_c_tokens, stage_shared_tokens = [], [], []
        for i, (fd, fc, hstf, use_this) in enumerate(zip(dfs_stages, cir_stages, self.hstf_blocks, self.hstf_stage_flags)):
            if self.use_hstf and use_this:
                fd, fc, sd, sc, ss = hstf(fd, fc)
            else:
                sd = F.adaptive_avg_pool2d(fd, 1).flatten(1)
                sc = F.adaptive_avg_pool2d(fc, 1).flatten(1)
                min_dim = min(sd.shape[1], sc.shape[1], self.token_dim)
                sd = sd[:, :min_dim]
                sc = sc[:, :min_dim]
                ss = 0.5 * (sd + sc)
            new_d.append(fd)
            new_c.append(fc)
            stage_d_tokens.append(sd)
            stage_c_tokens.append(sc)
            stage_shared_tokens.append(ss)
        return new_d, new_c, stage_d_tokens, stage_c_tokens, stage_shared_tokens

    def _motion_prototype(self, fd_final: torch.Tensor, t_size: int = 64) -> torch.Tensor:
        # pool over velocity axis -> keep time axis
        x = fd_final.mean(dim=2)  # [B, C, W]
        x = self.motion_head(x).squeeze(1)  # [B, W]
        x = F.interpolate(x.unsqueeze(1), size=t_size, mode='linear', align_corners=False).squeeze(1)
        return x

    def _range_prototype(self, fc_final: torch.Tensor, t_size: int = 64) -> torch.Tensor:
        # soft attention over distance axis
        energy = fc_final.mean(dim=1)  # [B, H, W]
        attn = torch.softmax(energy, dim=1)
        pos = torch.linspace(0.0, 1.0, energy.shape[1], device=energy.device).view(1, -1, 1)
        proto = (attn * pos).sum(dim=1)  # [B, W]
        proto = F.interpolate(proto.unsqueeze(1), size=t_size, mode='linear', align_corners=False).squeeze(1)
        return proto

    def forward(self, dfs: torch.Tensor, cir: torch.Tensor) -> Dict[str, Any]:
        dfs_stages = self.dfs_encoder.forward_stages(dfs)
        cir_stages = self.cir_encoder.forward_stages(cir)
        dfs_stages, cir_stages, stage_d, stage_c, stage_s = self._apply_hstf(dfs_stages, cir_stages)

        fd_final = dfs_stages[-1]
        fc_final = cir_stages[-1]

        h_d = self._gap(fd_final)
        h_c = self._gap(fc_final)
        h_s = stage_s[-1]

        z_d = self.d_proj(h_d)
        z_c = self.c_proj(h_c)

        aux_d2c = self.aux_d2c(z_d)
        aux_c2d = self.aux_c2d(z_c)

        p_v = self._motion_prototype(fd_final)
        p_r = self._range_prototype(fc_final)

        fused = torch.cat([h_d, h_c, h_s], dim=1)

        return {
            'h_d': h_d,
            'h_c': h_c,
            'h_s': h_s,
            'z_d': z_d,
            'z_c': z_c,
            'fused': fused,
            'stage_d': stage_d,
            'stage_c': stage_c,
            'stage_s': stage_s,
            'p_v': p_v,
            'p_r': p_r,
        }
