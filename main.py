"""
main.py - Unified CLI Entrypoint for Skin-Lesion Classification.
Direct execution without web server. Supports:
  1. Predict a single image: python main.py --image <image_path>
  2. Quick demo on sample:   python main.py
  3. Evaluate on test set:   python main.py --test
  4. Train a new model:      python main.py --train --epochs 15 --weighted_sampler
"""

import sys
import random
import argparse
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from data import CLASS_NAMES, CLASS_FULL_NAMES, get_transforms
from models.resnet import get_model
from evaluate import run_evaluation, resolve_checkpoint
from train import train_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_image(image_path: Path, checkpoint_path: Path = None):
    """
    Predict skin lesion class from a JPG image and assess clinical risk category.
    """
    if not image_path.exists():
        print(f"[!] Image file not found: {image_path}")
        return

    resolved_ckpt = resolve_checkpoint(str(checkpoint_path) if checkpoint_path else None)
    model = get_model(num_classes=7, pretrained=False).to(DEVICE)
    ckpt = torch.load(resolved_ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = get_transforms(is_train=False)
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_idx = int(probs.argmax())
    top_class = CLASS_NAMES[top_idx]
    top_prob = probs[top_idx] * 100.0

    # Categorize clinical risk level
    mel_p = probs[CLASS_NAMES.index("mel")] * 100.0
    bcc_p = probs[CLASS_NAMES.index("bcc")] * 100.0
    if top_class in ["mel", "bcc"] or mel_p > 20.0 or bcc_p > 25.0:
        risk = "HIGH RISK"
    elif top_class == "akiec":
        risk = "MODERATE RISK"
    else:
        risk = "LOW RISK"

    print("\n" + "=" * 74)
    print("      SKIN LESION DIAGNOSIS RESULT (HAM10000)")
    print("=" * 74)
    print(f"[*] Image File    : {image_path.name}")
    print(f"[*] Checkpoint    : {resolved_ckpt}")
    print(f"[*] Device        : {DEVICE}")
    print("-" * 74)
    print("CLASS PROBABILITY DISTRIBUTION:")
    for idx in probs.argsort()[::-1]:
        cname = CLASS_NAMES[idx]
        p = probs[idx] * 100.0
        bar = "#" * int(p // 4)
        print(f"  [{cname.upper():<5}] {CLASS_FULL_NAMES[cname]:<42} : {p:5.2f}% | {bar}")
    print("-" * 74)
    print(f">> TOP DIAGNOSIS  : {top_class.upper()} - {CLASS_FULL_NAMES[top_class]} ({top_prob:.2f}%)")
    print(f">> RISK ASSESSMENT: {risk}")
    print("=" * 74 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Skin-Lesion Classification CLI System")
    parser.add_argument("--image", type=str, default=None, help="Path to JPG image file for diagnosis")
    parser.add_argument("--test", action="store_true", help="Run benchmark evaluation on independent test set")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal", "weighted_ce"], help="Loss function")
    parser.add_argument("--use_alpha", action="store_true", help="Enable alpha class weights for Focal Loss")
    parser.add_argument("--weighted_sampler", action="store_true", help="Enable WeightedRandomSampler")
    parser.add_argument("--aug", action="store_true", help="Enable advanced data augmentation")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers")
    parser.add_argument("--save_dir", type=str, default="results/new_run", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pth file")
    args = parser.parse_args()

    if args.test:
        run_evaluation(Path(args.checkpoint) if args.checkpoint else None)
    elif args.train:
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
    else:
        if args.image:
            img_path = Path(args.image)
        else:
            samples = list(Path("sample_images").glob("*.jpg"))
            if not samples:
                samples = list(Path("data/images").glob("*.jpg"))
            if not samples:
                print("[!] No sample images found for prediction.")
                return
            img_path = random.choice(samples)
            print(f"[*] Automatically selected sample image: {img_path.name}")
        predict_image(img_path, Path(args.checkpoint) if args.checkpoint else None)


if __name__ == "__main__":
    main()
