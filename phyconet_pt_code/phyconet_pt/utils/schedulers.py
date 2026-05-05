from __future__ import annotations

import math
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_cosine_scheduler(
    optimizer: Optimizer,
    total_epochs: int,
    warmup_epochs: int = 0,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
