"""Train the FP32 MobileNetV3-UNet on the synthetic dataset.
    python -m edgeavi.ml.train --epochs 15 --batch 16
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from edgeavi.config import load_config
from edgeavi.ml.dataset import DefectDataset
from edgeavi.ml.model import MobileNetV3UNet


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean()


@torch.no_grad()
def validate(model, loader, dev):
    model.eval()
    ious = []
    for x, y in loader:
        pred = (torch.sigmoid(model(x.to(dev))) > 0.5).float().cpu()
        inter = (pred * y).sum((1, 2, 3))
        union = ((pred + y) > 0).float().sum((1, 2, 3))
        has_def = y.sum((1, 2, 3)) > 0
        ious.append((inter[has_def] / union[has_def].clamp(min=1)))   # IoU on defective samples
    return torch.cat(ious).mean().item() if ious else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    h, w = cfg["model"]["input_height"], cfg["model"]["input_width"]
    syn = Path(cfg["paths"]["synthetic_dir"])
    tr = DataLoader(DefectDataset(syn / "train", h, w, train=True), batch_size=a.batch, shuffle=True, num_workers=2, drop_last=True)
    va = DataLoader(DefectDataset(syn / "val", h, w), batch_size=a.batch, num_workers=2)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MobileNetV3UNet(pretrained=not a.no_pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    best = -1.0
    Path(cfg["model"]["checkpoint"]).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(5.0, device=dev)) + dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()
        miou = validate(model, va, dev)
        print(f"epoch {ep:02d} loss {total / len(tr):.4f}  val IoU(defective) {miou:.4f}")
        if miou > best:
            best = miou
            torch.save(model.state_dict(), cfg["model"]["checkpoint"])
    print(f"best val IoU {best:.4f} -> {cfg['model']['checkpoint']}")


if __name__ == "__main__":
    main()
