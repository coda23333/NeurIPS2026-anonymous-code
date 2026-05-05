from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from datasets import UltrasoundMatDataset, collate_pretrain
from models import PhyCoNetPT
from losses import bidirectional_info_nce, stagewise_contrastive, PhysicalConsistencyLoss
from utils.config import load_yaml
from utils.seed import set_seed
from utils.schedulers import build_cosine_scheduler
from utils.checkpoint import save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, default='')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    return parser.parse_args()


def make_loader(csv_path: str, cfg: Dict[str, Any], train: bool) -> DataLoader:
    ds = UltrasoundMatDataset(csv_path, cfg['preprocess'], supervised=False)
    return DataLoader(
        ds,
        batch_size=cfg['batch_size'],
        shuffle=train,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        drop_last=train,
        collate_fn=collate_pretrain,
    )


def build_optimizer(params, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    opt_cfg = cfg['optimizer']
    return torch.optim.AdamW(params, lr=opt_cfg['lr'], weight_decay=opt_cfg['weight_decay'])


def compute_pretrain_loss(outputs: Dict[str, Any], loss_cfg: Dict[str, Any], phy_loss_fn: PhysicalConsistencyLoss) -> tuple[torch.Tensor, Dict[str, float]]:
    temperature = float(loss_cfg.get('temperature', 0.07))
    total = torch.tensor(0.0, device=outputs['z_d'].device)
    logs: Dict[str, float] = {}

    if loss_cfg.get('use_global_contrast', True):
        l_global = bidirectional_info_nce(outputs['z_d'], outputs['z_c'], temperature=temperature)
        total = total + l_global
        logs['l_global'] = float(l_global.detach().item())
    else:
        logs['l_global'] = 0.0

    if loss_cfg.get('use_stage_contrast', True):
        l_stage = stagewise_contrastive(
            outputs['stage_d'], outputs['stage_c'], loss_cfg.get('stage_weights', [0.1, 0.2, 0.3, 0.4]), temperature=temperature
        )
        total = total + float(loss_cfg.get('lambda_stage', 0.5)) * l_stage
        logs['l_stage'] = float(l_stage.detach().item())
    else:
        logs['l_stage'] = 0.0

    if loss_cfg.get('use_phy_consistency', True):
        l_phy = phy_loss_fn(outputs['p_v'], outputs['p_r'])
        total = total + float(loss_cfg.get('lambda_phy', 0.2)) * l_phy
        logs['l_phy'] = float(l_phy.detach().item())
    else:
        logs['l_phy'] = 0.0

    logs['total'] = float(total.detach().item())
    return total, logs


def run_epoch(loader, model, optimizer, scaler, device, cfg, phy_loss_fn, train: bool):
    model.train(train)
    phy_loss_fn.train(train)
    loss_meter = {'total': 0.0, 'l_global': 0.0, 'l_stage': 0.0, 'l_phy': 0.0}
    n = 0

    pbar = tqdm(loader, desc='train' if train else 'val', leave=False)
    for batch in pbar:
        dfs = batch['dfs'].to(device, non_blocking=True)
        cir = batch['cir'].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with autocast(enabled=cfg.get('mixed_precision', True)):
                outputs = model(dfs, cir)
                loss, logs = compute_pretrain_loss(outputs, cfg['loss'], phy_loss_fn)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        bs = dfs.size(0)
        n += bs
        for k in loss_meter.keys():
            loss_meter[k] += logs[k] * bs
        pbar.set_postfix({k: f"{loss_meter[k] / max(n,1):.4f}" for k in ['total', 'l_global', 'l_stage']})

    return {k: v / max(n, 1) for k, v in loss_meter.items()}


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get('seed', 42)))
    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(args.train_csv, cfg, train=True)
    val_loader = make_loader(args.val_csv, cfg, train=False) if args.val_csv else None

    model = PhyCoNetPT(cfg['model']).to(device)
    phy_loss_fn = PhysicalConsistencyLoss().to(device)
    params = list(model.parameters())
    if cfg['loss'].get('use_phy_consistency', True):
        params += list(phy_loss_fn.parameters())
    optimizer = build_optimizer(params, cfg)
    scheduler = build_cosine_scheduler(
        optimizer,
        total_epochs=cfg['epochs'],
        warmup_epochs=cfg['scheduler'].get('warmup_epochs', 0),
        min_lr_ratio=cfg['scheduler'].get('min_lr', 1e-6) / max(cfg['optimizer']['lr'], 1e-12),
    )
    scaler = GradScaler(enabled=cfg.get('mixed_precision', True))

    best_metric = float('inf')
    for epoch in range(cfg['epochs']):
        train_logs = run_epoch(train_loader, model, optimizer, scaler, device, cfg, phy_loss_fn, train=True)
        if val_loader is not None:
            val_logs = run_epoch(val_loader, model, optimizer, scaler, device, cfg, phy_loss_fn, train=False)
            monitor = val_logs['total']
        else:
            val_logs = None
            monitor = train_logs['total']

        scheduler.step()

        state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'phy_loss_fn': phy_loss_fn.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': cfg,
            'train_logs': train_logs,
            'val_logs': val_logs,
        }

        save_checkpoint(state, save_dir / 'last.pt')
        if (epoch + 1) % int(cfg.get('save_every', 10)) == 0:
            save_checkpoint(state, save_dir / f'epoch_{epoch+1}.pt')
        if monitor < best_metric:
            best_metric = monitor
            save_checkpoint(state, save_dir / 'best.pt')

        msg = f"Epoch {epoch+1}/{cfg['epochs']} | train_total={train_logs['total']:.4f}"
        if val_logs is not None:
            msg += f" | val_total={val_logs['total']:.4f}"
        print(msg)


if __name__ == '__main__':
    main()
