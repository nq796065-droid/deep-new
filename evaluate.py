"""
evaluate.py - Evaluation script for HAM10000 Test Set.
Computes: Overall Accuracy, Balanced Accuracy, Macro F1, Weighted F1,
Melanoma Recall, Macro OvR ROC-AUC, Per-Class Breakdown, and Confusion Matrix.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix, roc_auc_score
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from data import get_dataloaders, CLASS_NAMES, CLASS_FULL_NAMES
from models.resnet import get_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = list(range(len(CLASS_NAMES)))


def find_default_checkpoint() -> Path:
    """Automatically detect best available checkpoint in results directory."""
    candidates = [
        Path("results/weighted_sampling_model.pth"),
        Path("results/weighted_sampling/best_model.pth"),
        Path("results/best_model.pth"),
        Path("results/baseline_model.pth")
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path("results/weighted_sampling_model.pth")


def resolve_checkpoint(name_or_path: str = None) -> Path:
    """
    Find checkpoint by path or experiment name.
    If none provided, safely fallback to available checkpoint.
    If specific name provided but missing, raise FileNotFoundError.
    """
    if name_or_path is None or str(name_or_path).strip() == "":
        default_ckpt = find_default_checkpoint()
        if not default_ckpt.exists():
            raise FileNotFoundError(
                "No default checkpoint found in results/ directory. "
                "Please specify a path using --checkpoint."
            )
        return default_ckpt

    p = Path(name_or_path)
    if p.exists():
        return p

    for candidate in [
        Path(f"results/{name_or_path}_model.pth"),
        Path(f"results/{name_or_path}/best_model.pth"),
        Path(f"results/{name_or_path}.pth")
    ]:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Checkpoint file not found: '{name_or_path}'. "
        f"Please check the path in results/ directory."
    )


def run_evaluation(checkpoint_path: Path = None, save_cm: bool = True):
    resolved_ckpt = resolve_checkpoint(str(checkpoint_path) if checkpoint_path else None)

    if resolved_ckpt.stem in ["best_model", "best"]:
        model_name = resolved_ckpt.parent.name
    else:
        model_name = resolved_ckpt.stem.replace("_model", "")

    print("\n" + "=" * 76)
    print("      OFFICIAL BENCHMARK TEST SET EVALUATION (HAM10000)")
    print("=" * 76)
    print(f"[*] Checkpoint : {resolved_ckpt}")
    print(f"[*] Config     : {model_name}")
    print(f"[*] Device     : {DEVICE}")

    _, _, test_loader = get_dataloaders()
    model = get_model(num_classes=7, pretrained=False).to(DEVICE)
    ckpt = torch.load(resolved_ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for images, targets in test_loader:
            outputs = model(images.to(DEVICE))
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)

    try:
        macro_auc = roc_auc_score(y_true, y_prob, labels=LABELS, multi_class="ovr", average="macro")
    except Exception:
        macro_auc = None

    p_cls = precision_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    r_cls = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    f1_cls = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)

    mel_idx = CLASS_NAMES.index("mel")
    mel_recall = r_cls[mel_idx]

    print("-" * 76)
    print("OVERALL BENCHMARK EVALUATION METRICS:")
    print(f"  * Overall Accuracy       : {acc * 100:6.2f}%")
    print(f"  * Balanced Accuracy      : {bal_acc * 100:6.2f}%")
    print(f"  * Macro F1-score (KEY)   : {macro_f1:6.4f}")
    print(f"  * Weighted F1-score      : {weighted_f1:6.4f}")
    if macro_auc is not None:
        print(f"  * Macro OvR ROC-AUC      : {macro_auc:6.4f}")
    print(f"  * Melanoma (MEL) Recall  : {mel_recall * 100:6.2f}%   [CRITICAL MINORITY SENSITIVITY]")
    print(f"  * Macro Recall           : {macro_rec * 100:6.2f}%")
    print(f"  * Macro Precision        : {macro_prec * 100:6.2f}%")
    print("-" * 76)
    print("PER-CLASS PERFORMANCE BREAKDOWN (ALL 7 CLASSES):")
    print(f"{'Code':<7} {'Diagnostic Category':<28} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support'}")
    print("-" * 76)
    for idx, c in enumerate(CLASS_NAMES):
        support = int((y_true == idx).sum())
        name = CLASS_FULL_NAMES[c][:26]
        print(f"{c.upper():<7} {name:<28} {p_cls[idx]*100:6.2f}%    {r_cls[idx]*100:6.2f}%    {f1_cls[idx]:6.4f}     {support:<5}")
    print("=" * 76)

    if save_cm:
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        plt.figure(figsize=(8, 6.5))
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

        sns.heatmap(
            cm_norm, annot=annot, fmt="", cmap="Blues",
            xticklabels=[c.upper() for c in CLASS_NAMES],
            yticklabels=[c.upper() for c in CLASS_NAMES]
        )
        plt.title(f"Test Confusion Matrix ({model_name})", fontweight="bold", pad=10)
        plt.ylabel("Ground Truth Class", fontweight="bold")
        plt.xlabel("Predicted Class", fontweight="bold")
        plt.tight_layout()

        out_cm = Path("results") / f"confusion_matrix_{model_name}.png"
        out_cm.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_cm, dpi=300)
        plt.savefig(Path("results/confusion_matrix_test.png"), dpi=300)
        plt.close()
        print(f"[+] Confusion matrix saved at: {out_cm}\n")

    return {
        "accuracy": acc, "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "macro_recall": macro_rec, "macro_precision": macro_prec,
        "melanoma_recall": mel_recall, "macro_auc": macro_auc
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official Test Set Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pth file")
    args = parser.parse_args()
    run_evaluation(Path(args.checkpoint) if args.checkpoint else None)
