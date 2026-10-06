"""
train.py - Pipeline Huấn luyện Mô hình ResNet-50 trên HAM10000.
Tối ưu hóa với AdamW, Cosine Annealing LR và lưu checkpoint tốt nhất theo Macro F1.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from data import get_dataloaders, set_seed
from models.resnet import get_model
from losses import get_loss_fn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_grad_scaler():
    """Hỗ trợ tương thích ngược GradScaler giữa các phiên bản PyTorch (Torch 2.0 -> 2.6+)."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
    return torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))


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


def train_model(
    epochs: int = 15,
    lr: float = 1e-4,
    loss_type: str = "ce",
    use_alpha: bool = False,
    use_weighted_sampler: bool = False,
    use_aug: bool = False,
    num_workers: int = 2,
    save_dir: str = "results/exp"
):
    # Guard chống cân bằng lớp 2 lần (Double Balancing)
    if use_weighted_sampler and loss_type == "weighted_ce":
        raise ValueError(
            "CẢNH BÁO KHOA HỌC: Không thể dùng đồng thời 'use_weighted_sampler=True' và 'loss_type=weighted_ce'. "
            "Việc này sẽ làm bù trừ mất cân bằng 2 lần (Double Class Balancing), làm lệch nghiêm trọng phân phối xác suất!"
        )
    if use_weighted_sampler and loss_type == "focal" and use_alpha:
        raise ValueError(
            "CẢNH BẢO KHOA HỌC: Không thể kết hợp 'use_weighted_sampler=True' và 'use_alpha=True' cho Focal Loss. "
            "Việc này sẽ tính trọng số lớp 2 lần (Double Balancing)!"
        )

    set_seed(42)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(
        use_weighted_sampler=use_weighted_sampler,
        use_aug=use_aug,
        num_workers=num_workers
    )
    model = get_model(num_classes=7, pretrained=True).to(DEVICE)

    counts = [train_loader.dataset.labels.count(i) for i in range(7)]
    criterion = get_loss_fn(loss_type, class_counts=counts, use_alpha=use_alpha, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = get_grad_scaler()

    best_macro_f1 = 0.0
    history = []
    print(f"[*] Bắt đầu huấn luyện trên thiết bị: {DEVICE}")
    print(f"[*] Loss: {loss_type} (alpha={use_alpha}) | Weighted Sampler: {use_weighted_sampler} | Augmentation: {use_aug}")

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1
        })

        print(
            f"Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Loss: {tr_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:5.2f}% | "
            f"Val Macro F1: {val_f1:6.4f}"
        )

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            torch.save(
                {'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_macro_f1': val_f1},
                save_path / "best_model.pth"
            )
            print(f"  [+] Đã lưu checkpoint mới tốt nhất tại epoch {epoch} (Macro F1: {val_f1:.4f})")

    # Lưu lịch sử huấn luyện
    pd.DataFrame(history).to_csv(save_path / "training_history.csv", index=False)
    print(f"[*] Hoàn thành huấn luyện. Best Val Macro F1: {best_macro_f1:.4f}")
    print(f"[*] Lịch sử huấn luyện đã được lưu tại: {save_path / 'training_history.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện ResNet-50 trên HAM10000")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal", "weighted_ce"])
    parser.add_argument("--use_alpha", action="store_true", help="Bật alpha-weighting cho Focal Loss")
    parser.add_argument("--weighted_sampler", action="store_true")
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="results/new_exp")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        lr=args.lr,
        loss_type=args.loss,
        use_alpha=args.use_alpha,
        use_weighted_sampler=args.weighted_sampler,
        use_aug=args.aug,
        num_workers=args.num_workers,
        save_dir=args.save_dir
    )
