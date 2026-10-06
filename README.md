# Skin Lesion Classification under Severe Class Imbalance (HAM10000)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-HAM10000-green.svg)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Deep Learning framework for classifying pigmented skin lesions across 7 dermatological diagnostic categories on the HAM10000 dataset, specifically researching and resolving severe class imbalance (the majority class `nv` accounts for 66.95% while the rarest minority class `df` accounts for only 1.15%).

---

## 📌 Project Overview
- **Objective**: Develop a ResNet-50 architecture to classify dermatoscopic images, systematically experimenting with and comparing class-imbalance mitigation techniques, with primary evaluation on **Macro F1** and **Balanced Accuracy**.
- **Zero Data Leakage**: Lesion-aware partitioning into Train / Val / Test sets (70% - 15% - 15%) using `StratifiedGroupKFold` on `lesion_id`. This guarantees that images of the same lesion from the same patient never appear across multiple splits.
- **5 Systematic Ablation Configurations**:
  1. **Baseline**: Cross-Entropy Loss + Random Sampling + Basic Augmentation.
  2. **Sampling Ablation**: Weighted Random Sampler (inverse class frequency).
  3. **Loss Function Ablation**: Focal Loss ($\gamma = 2.0$).
  4. **Data Augmentation Ablation**: Color Jitter + Random Affine + Rotation.
  5. **Combined Strategy**: Weighted Sampling + Focal Loss + Data Augmentation.

---

## 📂 Project Structure
```text
skin-lesion-classification/
├── data/                       # HAM10000 image data and metadata files
│   ├── HAM10000_metadata.csv   # Clinical metadata (lesion_id, dx, age, sex, etc.)
│   ├── splits.csv              # Deterministic split mapping (Zero-Leakage Grouped Split)
│   └── images/                 # 10,015 dermatoscopic JPG images
├── models/
│   ├── __init__.py
│   └── resnet.py               # 7-class ResNet-50 classification architecture
├── results/                    # Model weight checkpoints and reporting artifacts
│   ├── weighted_sampling_model.pth # Best model checkpoint (Macro F1: 0.6905)
│   ├── baseline_model.pth
│   ├── focal_loss_model.pth
│   ├── augmentation_model.pth
│   ├── combined_model.pth
│   ├── comparison.csv          # Comprehensive evaluation comparison across 5 configurations
│   ├── comparison_chart.png    # Metric comparison bar chart
│   ├── per_class_recall_comparison.png # Per-class sensitivity / recall comparison
│   ├── confusion_matrix_test.png       # Test set confusion matrix
│   ├── report.pdf              # Comprehensive academic report (11 pages)
│   └── ppt.pptx                # Presentation slide deck (4 structured slides)
├── data.py                     # Dataset management, Transforms, Grouped K-Fold split
├── losses.py                   # Loss function implementations (Cross-Entropy, Focal Loss, Class-Balanced)
├── train.py                    # Training pipeline, learning rate scheduling, checkpointing
├── evaluate.py                 # Comprehensive benchmark evaluation on the 1,431 test images
├── main.py                     # Main entry point (inference, evaluation, and training)
├── DATA.md                     # Dataset specification and zero-leakage protocol documentation
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation and benchmark overview
```

---

## ⚙️ Installation

```bash
# 1. Create and activate a Python virtual environment (Python 3.10+)
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start & Usage

### 1. Benchmark Evaluation on Test Set
Evaluate the best model checkpoint (`Weighted Sampling`) on all 1,431 independent test images (leakage-free). This generates a summary metric table, per-class breakdown, and saves the confusion matrix:
```bash
python evaluate.py
# Or run via main:
python main.py --test
```
*(Optionally specify an alternative checkpoint)*:
```bash
python evaluate.py results/baseline_model.pth
```

### 2. Direct Single-Image Inference
Pass any skin lesion image file path; the script will preprocess the image, compute full 7-class probability distributions, output the top diagnosis, and provide a clinical risk tier:
```bash
# Infer a specific image:
python main.py --image sample_images/mel_ISIC_0025964.jpg

# Or run a quick check on a random sample:
python main.py
```

### 3. Model Training
```bash
# Train with weighted sampling:
python train.py --epochs 15 --weighted_sampler

# Or via main:
python main.py --train --epochs 15
```

---

## 📈 Benchmark Results

Extracted directly from benchmark test evaluations on trained checkpoints (`results/comparison.csv`):

| Experimental Configuration | Loss Function | Sampling Strategy | Augmentation Strategy | Accuracy | Balanced Acc | Macro F1 (Primary) | Weighted F1 | Macro Recall |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | Cross-Entropy | Random | Basic Flip | 83.23% | 58.29% | 0.6197 | 0.8221 | 58.29% |
| **Weighted Sampling** | **Cross-Entropy** | **Weighted Random** | **Basic Flip** | **78.48%** | **68.86%** | **0.6905** | **0.7996** | **68.86%** |
| **Focal Loss** | Focal ($\gamma=2$) | Random | Basic Flip | 79.59% | 65.64% | 0.6550 | 0.8028 | 65.64% |
| **Data Augmentation** | Cross-Entropy | Random | Color + Affine | 81.48% | 57.59% | 0.5922 | 0.8033 | 57.59% |
| **Combined Strategy** | Focal ($\gamma=2$) | Weighted Random | Color + Affine | 68.48% | 67.61% | 0.5845 | 0.7157 | 67.61% |

### Key Experimental Insights:
1. **The Standard Accuracy Trap**: The Baseline configuration achieved high standard accuracy (83.23%) largely by over-predicting the majority class (`nv`, 67% of data). However, its Balanced Accuracy is only 58.29%, severely missing critical malignant lesions (`mel`, `bcc`).
2. **Superiority of Weighted Sampling**: Achieved the **highest Macro F1 (0.6905)** and **highest Balanced Accuracy (68.86%)**, boosting sensitivity for Basal Cell Carcinoma (`bcc`) to **79.73%** and Melanoma (`mel`) to **66.67%**.

---

## 📄 Academic Report & Defense Slides
- **Academic Research Report (PDF)**: `results/report.pdf` (11 pages, complete mathematical formulation, benchmark tables, and clinical error analysis).
- **Presentation Deck (PPTX)**: `results/ppt.pptx` (4 structured slides formatted for project defense).
