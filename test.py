"""
test.py - Script kiem thu nhanh model chuan de bai (HAM10000).
Chay 1 lenh duy nhat: python test.py
"""
import sys
from pathlib import Path
from evaluate import run_evaluation, DEFAULT_CKPT

if __name__ == "__main__":
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CKPT
    run_evaluation(ckpt)
