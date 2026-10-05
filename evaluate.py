"""
evaluate.py - Danh gia mo hinh tren tap Test (HAM10000).
Tinh toan cac metrics: Accuracy, Balanced Accuracy, Macro F1, Recall, Precision va Confusion Matrix.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from data import get_dataloaders, CLASS_NAMES, CLASS_FULL_NAMES
from models.resnet import get_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CKPT = Path("results/weighted_sampling_model.pth")
if not DEFAULT_CKPT.exists():
    DEFAULT_CKPT = Path("results/weighted_sampling/best_model.pth")


def resolve_checkpoint(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    for candidate in [Path(f"results/{name_or_path}_model.pth"), Path(f"results/{name_or_path}/best_model.pth"), Path(f"results/{name_or_path}.pth")]:
        if candidate.exists():
            return candidate
    return DEFAULT_CKPT


def run_evaluation(checkpoint_path: Path = DEFAULT_CKPT, save_cm: bool = True):
    checkpoint_path = resolve_checkpoint(str(checkpoint_path))
    print("\n" + "=" * 74)
    print("      OFFICIAL BENCHMARK TEST SET EVALUATION (HAM10000)")
    print("=" * 74)
    print(f"[*] Checkpoint : {checkpoint_path}")
    print(f"[*] Device     : {DEVICE}")

    _, _, test_loader = get_dataloaders()
    model = get_model(num_classes=7, pretrained=False).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            outputs = model(images.to(DEVICE))
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.numpy())

    y_true, y_pred = np.array(all_targets), np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)

    p_cls = precision_score(y_true, y_pred, average=None, zero_division=0)
    r_cls = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_cls = f1_score(y_true, y_pred, average=None, zero_division=0)

    print("-" * 74)
    print("TONG HOP CHI SO DANH GIA CHINH (MAIN EVALUATION METRICS):")
    print(f"  * Overall Accuracy       : {acc * 100:6.2f}%   (Tong do chinh xac toan bo)")
    print(f"  * Balanced Accuracy      : {bal_acc * 100:6.2f}%   (Trung binh Recall cac lop)")
    print(f"  * Macro F1-score (KEY)   : {macro_f1:6.4f}    (Chi so then chot theo de bai)")
    print(f"  * Weighted F1-score      : {weighted_f1:6.4f}")
    print(f"  * Macro Recall           : {macro_rec * 100:6.2f}%")
    print(f"  * Macro Precision        : {macro_prec * 100:6.2f}%")
    print("-" * 74)
    print("CHI TIET TUNG LOP TOAN BO 7 BENH (PER-CLASS PERFORMANCE BREAKDOWN):")
    print(f"{'Code':<7} {'Diagnostic Category':<28} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support'}")
    print("-" * 74)
    for idx, c in enumerate(CLASS_NAMES):
        support = int((y_true == idx).sum())
        name = CLASS_FULL_NAMES[c][:26]
        print(f"{c.upper():<7} {name:<28} {p_cls[idx]*100:6.2f}%    {r_cls[idx]*100:6.2f}%    {f1_cls[idx]:6.4f}     {support:<5}")
    print("=" * 74)

    if save_cm:
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        plt.figure(figsize=(8, 6.5))
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

        sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues",
                    xticklabels=[c.upper() for c in CLASS_NAMES],
                    yticklabels=[c.upper() for c in CLASS_NAMES])
        plt.title(f"Test Confusion Matrix ({checkpoint_path.parent.name})", fontweight="bold", pad=10)
        plt.ylabel("Ground Truth Class", fontweight="bold")
        plt.xlabel("Predicted Class", fontweight="bold")
        plt.tight_layout()
        out_cm = Path("results/confusion_matrix_test.png")
        out_cm.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_cm, dpi=300)
        plt.close()
        print(f"[+] Ma tran nham lan da duoc luu tai: {out_cm}\n")

    return {
        "accuracy": acc, "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "macro_recall": macro_rec, "macro_precision": macro_prec
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official Test Set Evaluation")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    args = parser.parse_args()
    run_evaluation(Path(args.checkpoint))
