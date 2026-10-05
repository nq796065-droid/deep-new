"""
main.py - Chuong trinh Thuc thi Chinh (Unified CLI Entrypoint).
Chay truc tiep, khong can web server. Ho tro:
  - Du doan anh:        python main.py --image <duong_dan_anh>
  - Kiem thu tap Test:  python main.py --test
  - Huan luyen mo hinh: python main.py --train --epochs 15
Tong cong: ~45 dong code.
"""

import argparse
from pathlib import Path

from predict import predict_image
from evaluate import run_evaluation, DEFAULT_CKPT
from train import train_model


def main():
    parser = argparse.ArgumentParser(description="Skin-Lesion Classification CLI System")
    parser.add_argument("--image", type=str, default=None, help="File anh JPG can du doan")
    parser.add_argument("--test", action="store_true", help="Kiem thu toan bo tap Test")
    parser.add_argument("--train", action="store_true", help="Huan luyen mo hinh moi")
    parser.add_argument("--epochs", type=int, default=15, help="So epoch neu huan luyen")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    args = parser.parse_args()

    if args.test:
        run_evaluation(Path(args.checkpoint))
    elif args.train:
        train_model(epochs=args.epochs, save_dir="results/new_run")
    else:
        if args.image:
            img_path = Path(args.image)
        else:
            samples = list(Path("sample_images").glob("*.jpg"))
            if not samples:
                samples = list(Path("data/images").glob("*.jpg"))
            if not samples:
                print("[!] Khong tim thay anh mau de du doan.")
                return
            import random
            img_path = random.choice(samples)
            print(f"[*] Tu dong chon anh mau kiem tra: {img_path.name}")
        predict_image(img_path, Path(args.checkpoint))


if __name__ == "__main__":
    main()
