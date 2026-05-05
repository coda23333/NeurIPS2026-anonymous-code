from __future__ import annotations

from typing import Dict, Any
import numpy as np


def _zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = x.mean()
    std = x.std()
    return (x - mu) / (std + eps)


def _minmax(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mn = x.min()
    mx = x.max()
    return (x - mn) / (mx - mn + eps)


def preprocess_dfs(dfs: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    x = dfs.astype(np.float32)
    eps = float(cfg.get('eps', 1e-6))
    if cfg.get('dfs_log', True):
        x = np.log1p(np.maximum(x, 0.0))
    norm_mode = cfg.get('normalize', 'per_sample_zscore')
    if norm_mode == 'per_sample_zscore':
        x = _zscore(x, eps)
    elif norm_mode == 'minmax':
        x = _minmax(x, eps)
    return x.astype(np.float32)


def preprocess_cir(cir: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    x = cir
    eps = float(cfg.get('eps', 1e-6))
    if np.iscomplexobj(x) and cfg.get('cir_abs', True):
        x = np.abs(x)
    x = x.astype(np.float32)
    if cfg.get('cir_log', True):
        x = np.log1p(np.maximum(x, 0.0))
    norm_mode = cfg.get('normalize', 'per_sample_zscore')
    if norm_mode == 'per_sample_zscore':
        x = _zscore(x, eps)
    elif norm_mode == 'minmax':
        x = _minmax(x, eps)
    return x.astype(np.float32)
