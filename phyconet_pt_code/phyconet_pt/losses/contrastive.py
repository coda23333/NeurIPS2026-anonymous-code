from __future__ import annotations

from typing import List
import torch
import torch.nn.functional as F


def cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    return a @ b.t()


def bidirectional_info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits_ab = cosine_sim_matrix(a, b) / temperature
    logits_ba = cosine_sim_matrix(b, a) / temperature
    targets = torch.arange(a.size(0), device=a.device)
    loss_ab = F.cross_entropy(logits_ab, targets)
    loss_ba = F.cross_entropy(logits_ba, targets)
    return 0.5 * (loss_ab + loss_ba)


def stagewise_contrastive(
    a_list: List[torch.Tensor],
    b_list: List[torch.Tensor],
    stage_weights: List[float],
    temperature: float = 0.07,
) -> torch.Tensor:
    assert len(a_list) == len(b_list) == len(stage_weights)
    loss = 0.0
    total_w = 0.0
    for a, b, w in zip(a_list, b_list, stage_weights):
        if a.shape[1] != b.shape[1]:
            d = min(a.shape[1], b.shape[1])
            a = a[:, :d]
            b = b[:, :d]
        loss = loss + w * bidirectional_info_nce(a, b, temperature)
        total_w += w
    return loss / max(total_w, 1e-8)
