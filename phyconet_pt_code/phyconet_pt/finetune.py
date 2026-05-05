from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from datasets import UltrasoundMatDataset, collate_supervised
from models import PhyCoNetPT, DownstreamClassifier
from losses import bidirectional_info_nce, PhysicalConsistencyLoss
from utils.config import load_yaml
from utils.seed import set_seed
from utils.schedulers import build_cosine_scheduler
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import accuracy


class DownstreamModel(nn.Module):
    def __init__(self, backbone: PhyCoNetPT, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = DownstreamClassifier(backbone.out_dim, num_classes=num_classes)

    def forward(self, dfs: torch.Tensor, cir: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.backbone(dfs, cir)
        logits = self.classifier(out['fused'])
        out['logits'] = logits
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--pretrained_ckpt', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--mode', type=str, choices=['frozen', 'finetune'], required=True)
    parser.add_argument('--config', type=str, required=True)
    return parser.parse_args()


def make_loader(csv_path: str, cfg: Dict[str, Any], train: bool) -> DataLoader:
    ds = UltrasoundMatDataset(csv_path, cfg['preprocess'], supervised=True)
    return DataLoader(
        ds,
        batch_size=cfg['batch_size'],
        shuffle=train,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_supervised,
    )


def configure_trainable_params(model: DownstreamModel, mode: str) -> None:
    if mode == 'frozen':
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif mode == 'finetune':
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f'Unknown mode: {mode}')


def build_optimizer(model: DownstreamModel, cfg: Dict[str, Any], mode: str) -> torch.optim.Optimizer:
    opt_cfg = cfg['optimizer']
    if mode == 'frozen':
        params = [{'params': model.classifier.parameters(), 'lr': opt_cfg['head_lr']}]
    else:
        params = [
            {'params': model.backbone.parameters(), 'lr': opt_cfg['backbone_lr']},
            {'params': model.classifier.parameters(), 'lr': opt_cfg['head_lr']},
        ]
    return torch.optim.AdamW(params, weight_decay=opt_cfg['weight_decay'])


def compute_loss(outputs: Dict[str, torch.Tensor], labels: torch.Tensor, cfg: Dict[str, Any], phy_loss_fn: PhysicalConsistencyLoss) -> tuple[torch.Tensor, Dict[str, float]]:
    ce = nn.CrossEntropyLoss(label_smoothing=float(cfg.get('label_smoothing', 0.0)))(outputs['logits'], labels)
    total = ce
    logs = {'ce': float(ce.detach().item()), 'l_reg_global': 0.0, 'l_reg_phy': 0.0}

    ft_cfg = cfg['finetune']
    if ft_cfg.get('use_reg_global', False):
        l_global = bidirectional_info_nce(outputs['z_d'], outputs['z_c'])
        total = total + float(ft_cfg.get('beta_global', 0.05)) * l_global
        logs['l_reg_global'] = float(l_global.detach().item())

    if ft_cfg.get('use_reg_phy', False):
        l_phy = phy_loss_fn(outputs['p_v'], outputs['p_r'])
        total = total + float(ft_cfg.get('beta_phy', 0.02)) * l_phy
        logs['l_reg_phy'] = float(l_phy.detach().item())

    logs['total'] = float(total.detach().item())
    return total, logs


def run_epoch(loader, model, optimizer, scaler, device, cfg, phy_loss_fn, train: bool):
    model.train(train)
    if not train and cfg['finetune'].get('use_reg_phy', False):
        phy_loss_fn.eval()
    else:
        phy_loss_fn.train(train and cfg['finetune'].get('use_reg_phy', False))
    loss_meter = {'total': 0.0, 'ce': 0.0, 'l_reg_global': 0.0, 'l_reg_phy': 0.0, 'acc': 0.0}
    n = 0

    pbar = tqdm(loader, desc='train' if train else 'val', leave=False)
    for batch in pbar:
        dfs = batch['dfs'].to(device, non_blocking=True)
        cir = batch['cir'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with autocast(enabled=cfg.get('mixed_precision', True)):
                outputs = model(dfs, cir)
                loss, logs = compute_loss(outputs, labels, cfg, phy_loss_fn)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        bs = dfs.size(0)
        acc = accuracy(outputs['logits'].detach(), labels)
        n += bs
        loss_meter['acc'] += acc * bs
        for k in ['total', 'ce', 'l_reg_global', 'l_reg_phy']:
            loss_meter[k] += logs[k] * bs
        pbar.set_postfix({'acc': f"{loss_meter['acc']/max(n,1):.4f}", 'loss': f"{loss_meter['total']/max(n,1):.4f}"})

    return {k: v / max(n, 1) for k, v in loss_meter.items()}


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg['finetune']['mode'] = args.mode
    set_seed(int(cfg.get('seed', 42)))
    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(args.train_csv, cfg, train=True)
    val_loader = make_loader(args.val_csv, cfg, train=False)

    ckpt = load_checkpoint(args.pretrained_ckpt, map_location='cpu')
    pre_cfg = ckpt['config']['model'] if 'config' in ckpt else {
        'proj_dim': 256,
        'token_dim': 128,
        'token_grid': [4, 4],
        'use_hstf': True,
        'hstf_stage_flags': [True, True, True, True],
        'use_temporal_mhsa_dfs': True,
        'use_local_attn_cir': True,
    }
    backbone = PhyCoNetPT(pre_cfg)
    backbone.load_state_dict(ckpt['model'], strict=True)
    model = DownstreamModel(backbone, num_classes=args.num_classes).to(device)

    phy_loss_fn = PhysicalConsistencyLoss().to(device)
    if 'phy_loss_fn' in ckpt:
        phy_loss_fn.load_state_dict(ckpt['phy_loss_fn'], strict=False)

    configure_trainable_params(model, args.mode)
    optimizer = build_optimizer(model, cfg, args.mode)
    base_lr = max(cfg['optimizer']['head_lr'], cfg['optimizer']['backbone_lr'])
    scheduler = build_cosine_scheduler(
        optimizer,
        total_epochs=cfg['epochs'],
        warmup_epochs=cfg['scheduler'].get('warmup_epochs', 0),
        min_lr_ratio=cfg['scheduler'].get('min_lr', 1e-6) / max(base_lr, 1e-12),
    )
    scaler = GradScaler(enabled=cfg.get('mixed_precision', True))

    best_acc = -1.0
    for epoch in range(cfg['epochs']):
        train_logs = run_epoch(train_loader, model, optimizer, scaler, device, cfg, phy_loss_fn, train=True)
        val_logs = run_epoch(val_loader, model, optimizer, scaler, device, cfg, phy_loss_fn, train=False)
        scheduler.step()

        state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'backbone': model.backbone.state_dict(),
            'classifier': model.classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': cfg,
            'train_logs': train_logs,
            'val_logs': val_logs,
        }
        save_checkpoint(state, save_dir / 'last.pt')
        if (epoch + 1) % int(cfg.get('save_every', 5)) == 0:
            save_checkpoint(state, save_dir / f'epoch_{epoch+1}.pt')
        if val_logs['acc'] > best_acc:
            best_acc = val_logs['acc']
            save_checkpoint(state, save_dir / 'best.pt')

        print(
            f"Epoch {epoch+1}/{cfg['epochs']} | "
            f"train_acc={train_logs['acc']:.4f} val_acc={val_logs['acc']:.4f} "
            f"train_loss={train_logs['total']:.4f} val_loss={val_logs['total']:.4f}"
        )


if __name__ == '__main__':
    main()
