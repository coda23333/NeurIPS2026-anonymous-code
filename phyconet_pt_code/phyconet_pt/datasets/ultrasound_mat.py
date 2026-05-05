from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from .preprocess import preprocess_dfs, preprocess_cir


@dataclass
class SampleItem:
    dfs: torch.Tensor
    cir: torch.Tensor
    label: Optional[torch.Tensor]
    path: str


class UltrasoundMatDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        preprocess_cfg: Dict[str, Any],
        supervised: bool,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        if 'path' not in self.df.columns:
            raise ValueError(f'{csv_path} must contain a path column.')
        if supervised and 'label' not in self.df.columns:
            raise ValueError(f'{csv_path} must contain a label column for supervised use.')
        self.preprocess_cfg = preprocess_cfg
        self.supervised = supervised

    def __len__(self) -> int:
        return len(self.df)

    def _load_mat(self, path: str | Path) -> Dict[str, Any]:
        return sio.loadmat(path)

    def __getitem__(self, index: int) -> SampleItem:
        row = self.df.iloc[index]
        path = str(row['path'])
        mat = self._load_mat(path)
        if 'dfs2d' not in mat:
            raise KeyError(f'dfs2d not found in {path}')
        if 'cir' not in mat:
            raise KeyError(f'cir not found in {path}')

        dfs = preprocess_dfs(mat['dfs2d'], self.preprocess_cfg)
        cir = preprocess_cir(mat['cir'], self.preprocess_cfg)

        dfs_t = torch.from_numpy(dfs).unsqueeze(0).float()
        cir_t = torch.from_numpy(cir).unsqueeze(0).float()

        label_t = None
        if self.supervised:
            label_t = torch.tensor(int(row['label']), dtype=torch.long)

        return SampleItem(dfs=dfs_t, cir=cir_t, label=label_t, path=path)


def collate_pretrain(batch: list[SampleItem]) -> Dict[str, torch.Tensor | list[str]]:
    dfs = torch.stack([b.dfs for b in batch], dim=0)
    cir = torch.stack([b.cir for b in batch], dim=0)
    paths = [b.path for b in batch]
    return {'dfs': dfs, 'cir': cir, 'paths': paths}


def collate_supervised(batch: list[SampleItem]) -> Dict[str, torch.Tensor | list[str]]:
    dfs = torch.stack([b.dfs for b in batch], dim=0)
    cir = torch.stack([b.cir for b in batch], dim=0)
    labels = torch.stack([b.label for b in batch if b.label is not None], dim=0)
    paths = [b.path for b in batch]
    return {'dfs': dfs, 'cir': cir, 'labels': labels, 'paths': paths}
