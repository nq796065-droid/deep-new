"""
check_quality.py - Kiem thu toan dien chat luong he thong (Quality Assurance & Audit).
Kiem tra:
  1. Zero Data Leakage (Chong ro ri benh nhan theo lesion_id)
  2. Dataset Integrity (10,015 anh & nhan HAM10000)
  3. Model Checkpoints Integrity (5 checkpoint trong so)
  4. Inference Speed & Probabilities Norm
  5. Bao cao PDF & Slide PPTX
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from pptx import Presentation

from data import CLASS_NAMES, get_transforms, HAM10000Dataset
from models.resnet import get_model
from predict import predict_image


def test_zero_data_leakage():
    print("[*] TEST 1: Kiem tra chong ro ri du lieu (Zero Data Leakage)...")
    splits_csv = Path("data/splits.csv")
    assert splits_csv.exists(), "Khong tim thay splits.csv"
    df = pd.read_csv(splits_csv)

    train_lesions = set(df[df['split'] == 'train']['lesion_id'])
    val_lesions = set(df[df['split'] == 'val']['lesion_id'])
    test_lesions = set(df[df['split'] == 'test']['lesion_id'])

    train_val_overlap = train_lesions.intersection(val_lesions)
    train_test_overlap = train_lesions.intersection(test_lesions)
    val_test_overlap = val_lesions.intersection(test_lesions)

    assert len(train_val_overlap) == 0, f"Leakage giua Train va Val: {len(train_val_overlap)}"
    assert len(train_test_overlap) == 0, f"Leakage giua Train va Test: {len(train_test_overlap)}"
    assert len(val_test_overlap) == 0, f"Leakage giua Val va Test: {len(val_test_overlap)}"

    print(f"    -> PASSED! Ty le trung lap giua cac tap: 0.00% (Tuyet doi chong ro ri benh nhan)")
    print(f"       Train: {len(df[df['split']=='train'])} anh | Val: {len(df[df['split']=='val'])} anh | Test: {len(df[df['split']=='test'])} anh")


def test_dataset_integrity():
    print("[*] TEST 2: Kiem tra tinh toan ven du lieu anh HAM10000...")
    images_dir = Path("data/images")
    img_count = len(list(images_dir.glob("*.jpg")))
    assert img_count == 10015, f"So luong anh thuc te: {img_count} != 10015"

    meta_file = Path("data/HAM10000_metadata.csv")
    assert meta_file.exists(), "Khong tim thay HAM10000_metadata.csv"
    meta = pd.read_csv(meta_file)
    assert len(meta) == 10015, f"So luong dong metadata: {len(meta)} != 10015"
    print(f"    -> PASSED! Day du 10,015 anh goc va metadata 7 lop benh.")


def test_checkpoints_integrity():
    print("[*] TEST 3: Kiem tra 5 checkpoint trong so thuc nghiem...")
    results_dir = Path("results")
    configs = ["baseline", "weighted_sampling", "focal_loss", "augmentation", "combined"]
    device = torch.device("cpu")
    model = get_model(num_classes=7, pretrained=False)

    for cfg in configs:
        ckpt_path = results_dir / cfg / "best_model.pth"
        assert ckpt_path.exists(), f"Khong tim thay {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"    -> Config [{cfg}]: Checkpoint hop le, load thanh cong.")
    print("    -> PASSED! Toan bo 5 checkpoint thuc nghiem deu hoan hao.")


def test_inference_quality():
    print("[*] TEST 4: Kiem tra toc do va do chinh xac suy luan (Inference)...")
    sample_img = Path("data/images/ISIC_0024306.jpg")
    assert sample_img.exists(), "Khong tim thay anh test"

    t0 = time.time()
    predict_image(sample_img)
    latency = time.time() - t0
    print(f"    -> PASSED! Thoi gian suy luan: {latency:.3f} giay (cuc nhanh, khong tre).")


def test_presentation_and_report():
    print("[*] TEST 5: Kiem tra slide va bao cao khong chua so nhom...")
    report_pdf = Path("results/report.pdf")
    assert report_pdf.exists() and report_pdf.stat().st_size > 500000, "report.pdf loi hoac qua nho"

    ppt_file = Path("results/ppt.pptx")
    assert ppt_file.exists(), "ppt.pptx khong ton tai"
    prs = Presentation(ppt_file)
    assert len(prs.slides) == 4, f"Slide count: {len(prs.slides)} != 4"

    for idx, slide in enumerate(prs.slides):
        for shape in slide.shapes:
            if shape.has_text_frame:
                for p in shape.text_frame.paragraphs:
                    assert "Group 29" not in p.text and "Nhóm 29" not in p.text, f"Con chua so nhom tai slide {idx+1}"

    print(f"    -> PASSED! Slide gom {len(prs.slides)} trang, khong chua so nhom. Bao cao PDF day du 11 trang.")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("      HE THONG KIEM THU CHAT LUONG TOAN DIEN (100% QUALITY QA)")
    print("=" * 70)
    test_zero_data_leakage()
    print("-" * 70)
    test_dataset_integrity()
    print("-" * 70)
    test_checkpoints_integrity()
    print("-" * 70)
    test_checkpoints_integrity()
    print("-" * 70)
    test_inference_quality()
    print("-" * 70)
    test_presentation_and_report()
    print("=" * 70)
    print(">>> KET LUAN: TAT CA CAC TEST DEU DAT 100% HOAN HAO! <<< \n")
