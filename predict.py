"""
predict.py - Du doan truc tiep anh ton thuong da (Khong can Server).
Cach dung:
  python predict.py --image data/images/ISIC_0024306.jpg
  python predict.py   (Tu dong test 1 anh mau ngau nhien)
Tong cong: ~85 dong code.
"""

import sys
import argparse
import random
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


def predict_image(image_path: Path, checkpoint_path: Path = DEFAULT_CKPT):
    if not image_path.exists():
        print(f"[!] Khong tim thay file anh: {image_path}")
        return

    checkpoint_path = resolve_checkpoint(str(checkpoint_path))
    # Load model
    model = get_model(num_classes=7, pretrained=False).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Tien xu ly anh
    transform = get_transforms(is_train=False)
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Du doan
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_idx = int(probs.argmax())
    top_class = CLASS_NAMES[top_idx]
    top_prob = probs[top_idx] * 100.0

    # Danh gia rui ro y khoa
    mel_p = probs[CLASS_NAMES.index("mel")] * 100.0
    bcc_p = probs[CLASS_NAMES.index("bcc")] * 100.0
    if top_class in ["mel", "bcc"] or mel_p > 20.0 or bcc_p > 25.0:
        risk = "HIGH RISK (Nghi ngo ac tinh - Khuyen nghi sinh thiet/chuyen khoa)"
    elif top_class == "akiec":
        risk = "MODERATE RISK (Tien ung thu day sung quang hoa - Can theo doi sat)"
    else:
        risk = "LOW RISK (Lanh tinh - Theo doi dinh ky theo quy tac ABCD)"

    print("\n" + "=" * 70)
    print("      KET QUA CHAN DOAN TON THUONG SAC TO DA (HAM10000)")
    print("=" * 70)
    print(f"[*] File anh      : {image_path.name}")
    print(f"[*] Checkpoint    : {checkpoint_path}")
    print(f"[*] Thiet bi      : {DEVICE}")
    print("-" * 70)
    print("PHAN PHOI XAC SUAT 7 NHOM BENH (CLASS PROBABILITIES):")
    for idx in probs.argsort()[::-1]:
        cname = CLASS_NAMES[idx]
        p = probs[idx] * 100.0
        bar = "#" * int(p // 4)
        print(f"  [{cname.upper():<5}] {CLASS_FULL_NAMES[cname]:<42} : {p:5.2f}% | {bar}")
    print("-" * 70)
    print(f">> CHAN DOAN HANG DAU: {top_class.upper()} - {CLASS_FULL_NAMES[top_class]} ({top_prob:.2f}%)")
    print(f">> MUC DO RUI RO     : {risk}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Du doan truc tiep 1 anh")
    parser.add_argument("--image", type=str, default=None, help="Duong dan den file anh JPG")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    args = parser.parse_args()

    if args.image:
        target = Path(args.image)
    else:
        samples = list(Path("sample_images").glob("*.jpg"))
        if not samples:
            samples = list(Path("data/images").glob("*.jpg"))
        target = random.choice(samples) if samples else Path("sample.jpg")
        print(f"[*] Tu dong chon ngau nhien anh mau: {target.name}")

    predict_image(target, Path(args.checkpoint))
