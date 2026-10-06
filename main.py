"""
main.py - Điểm thực thi chính hợp nhất (Unified CLI Entrypoint).
Chạy trực tiếp, không cần Web Server. Tích hợp toàn diện:
  1. Dự đoán trực tiếp 1 ảnh:  python main.py --image <duong_dan_anh>
  2. Dự đoán nhanh ảnh mẫu:    python main.py
  3. Kiểm thử trên tập Test:   python main.py --test
  4. Huấn luyện mô hình mới:   python main.py --train --epochs 15 --weighted_sampler
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
    Dự đoán tổn thương da từ file ảnh JPG và phân loại mức độ rủi ro sơ bộ.
    Lưu ý: Ngưỡng sàng lọc (mel > 20%, bcc > 25%) là ngưỡng heuristic sơ bộ,
    chưa qua hiệu chuẩn xác suất (probability calibration) trên tập validation,
    chỉ phục vụ mục đích minh họa nghiên cứu.
    """
    if not image_path.exists():
        print(f"[!] Không tìm thấy file ảnh: {image_path}")
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

    # Phân loại mức độ rủi ro theo ngưỡng heuristic
    mel_p = probs[CLASS_NAMES.index("mel")] * 100.0
    bcc_p = probs[CLASS_NAMES.index("bcc")] * 100.0
    if top_class in ["mel", "bcc"] or mel_p > 20.0 or bcc_p > 25.0:
        risk = "HIGH RISK (Nghi ngờ ác tính theo heuristic - Khuyến nghị sinh thiết/khám chuyên khoa)"
    elif top_class == "akiec":
        risk = "MODERATE RISK (Tiền ung thư dày sừng quang hóa - Cần theo dõi sát)"
    else:
        risk = "LOW RISK (Lành tính - Theo dõi định kỳ theo quy tắc ABCD)"

    print("\n" + "=" * 74)
    print("      KẾT QUẢ CHẨN ĐOÁN TỔN THƯƠNG SẮC TỐ DA (HAM10000)")
    print("=" * 74)
    print(f"[*] File ảnh      : {image_path.name}")
    print(f"[*] Checkpoint    : {resolved_ckpt}")
    print(f"[*] Thiết bị      : {DEVICE}")
    print("-" * 74)
    print("PHÂN PHỐI XÁC SUẤT 7 NHÓM BỆNH (CLASS PROBABILITIES):")
    for idx in probs.argsort()[::-1]:
        cname = CLASS_NAMES[idx]
        p = probs[idx] * 100.0
        bar = "#" * int(p // 4)
        print(f"  [{cname.upper():<5}] {CLASS_FULL_NAMES[cname]:<42} : {p:5.2f}% | {bar}")
    print("-" * 74)
    print(f">> CHẨN ĐOÁN HÀNG ĐẦU : {top_class.upper()} - {CLASS_FULL_NAMES[top_class]} ({top_prob:.2f}%)")
    print(f">> MỨC ĐỘ RỦI RO      : {risk}")
    print("-" * 74)
    print("[!] MIỄN TRỪ TRÁCH NHIỆM Y KHOA (MEDICAL DISCLAIMER):")
    print("    Kết quả do mô hình AI đưa ra chỉ mang tính chất hỗ trợ nghiên cứu và sàng lọc ban đầu,")
    print("    TUYỆT ĐỐI KHÔNG thay thế cho chẩn đoán mô bệnh học lâm sàng của bác sĩ chuyên khoa da liễu.")
    print("=" * 74 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Skin-Lesion Classification CLI System")
    parser.add_argument("--image", type=str, default=None, help="File ảnh JPG cần chẩn đoán")
    parser.add_argument("--test", action="store_true", help="Kiểm thử toàn bộ tập Test độc lập")
    parser.add_argument("--train", action="store_true", help="Huấn luyện mô hình mới")
    parser.add_argument("--epochs", type=int, default=15, help="Số epoch nếu huấn luyện")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate khi huấn luyện")
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal", "weighted_ce"], help="Hàm mất mát")
    parser.add_argument("--use_alpha", action="store_true", help="Bật alpha class weights cho Focal Loss")
    parser.add_argument("--weighted_sampler", action="store_true", help="Bật WeightedRandomSampler")
    parser.add_argument("--aug", action="store_true", help="Bật Data Augmentation nâng cao")
    parser.add_argument("--num_workers", type=int, default=2, help="Số luồng worker cho DataLoader")
    parser.add_argument("--save_dir", type=str, default="results/new_run", help="Thư mục lưu checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn file checkpoint .pth")
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
                print("[!] Không tìm thấy ảnh để dự đoán.")
                return
            img_path = random.choice(samples)
            print(f"[*] Tự động chọn ngẫu nhiên ảnh mẫu kiểm tra: {img_path.name}")
        predict_image(img_path, Path(args.checkpoint) if args.checkpoint else None)


if __name__ == "__main__":
    main()
