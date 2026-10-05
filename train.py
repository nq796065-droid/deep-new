"""
train.py - Pipeline Huan luyen Mo hinh ResNet-50 tren HAM10000.
Toi uu hoa voi AdamW, Cosine Annealing LR va luu checkpoint tot nhat theo Macro F1.
Tong cong: ~110 dong code.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from data import get_dataloaders
from models.resnet import get_model
from losses import FocalLoss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    return total_loss / len(loader), acc, macro_f1


def train_model(epochs: int = 15, lr: float = 1e-4, loss_type: str = "ce", use_weighted_sampler: bool = False, use_aug: bool = False, save_dir: str = "results/exp"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    train_loader, val_loader, _ = get_dataloaders(use_weighted_sampler=use_weighted_sampler, use_aug=use_aug)
    model = get_model(num_classes=7, pretrained=True).to(DEVICE)
    
    criterion = FocalLoss(gamma=2.0) if loss_type == "focal" else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

    best_macro_f1 = 0.0
    print(f"[*] Bat dau huan luyen tren {DEVICE} | Loss: {loss_type} | Weighted Sampler: {use_weighted_sampler}")

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        scheduler.step()

        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_macro_f1': val_f1}, save_path / "best_model.pth")
            print(f"  [+] Da luu best checkpoint moi tai epoch {epoch} (Macro F1: {val_f1:.4f})")

    print(f"[*] Hoan thanh huan luyen. Best Val Macro F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huan luyen ResNet-50")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--weighted_sampler", action="store_true")
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--save_dir", type=str, default="results/new_exp")
    args = parser.parse_args()

    train_model(epochs=args.epochs, loss_type=args.loss, use_weighted_sampler=args.weighted_sampler, use_aug=args.aug, save_dir=args.save_dir)
