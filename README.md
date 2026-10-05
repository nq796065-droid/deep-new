# Skin-Lesion Classification under Severe Class Imbalance (HAM10000)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-HAM10000-green.svg)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Hệ thống Deep Learning phân loại tổn thương sắc tố da trên 7 nhóm bệnh da liễu (tập dữ liệu HAM10000), nghiên cứu và khắc phục triệt để hiện tượng mất cân bằng dữ liệu nghiêm trọng (Severe Class Imbalance: lớp đa số NV chiếm 66.95%, lớp thiểu số DF chỉ 1.15%).

---

## 📌 Tổng Quan Đề Tài (Project Overview)
- **Mục tiêu**: Xây dựng mô hình ResNet-50 phân loại ảnh da liễu dermatoscopic, thực nghiệm và so sánh có hệ thống các kỹ thuật giải quyết mất cân bằng lớp, tập trung vào hai chỉ số then chốt là **Macro F1** và **Balanced Accuracy**.
- **Không rò rỉ dữ liệu (Zero Data Leakage)**: Phân tách tập Train / Val / Test (70% - 15% - 15%) dựa trên `lesion_id` qua `StratifiedGroupKFold`. Đảm bảo ảnh cùng một tổn thương của một bệnh nhân không bao giờ xuất hiện đồng thời ở nhiều tập.
- **5 Cấu hình Thực nghiệm Ablation**:
  1. **Baseline**: Cross-Entropy Loss + Random Sampling + Basic Augmentation.
  2. **Sampling Ablation**: Weighted Random Sampler (nghịch đảo tần suất lớp).
  3. **Loss Function Ablation**: Focal Loss ($\gamma = 2.0$).
  4. **Data Augmentation Ablation**: Color Jitter + Random Affine + Rotation.
  5. **Combined Strategy**: Kết hợp Weighted Sampling + Focal Loss + Data Augmentation.

---

## 📂 Cấu Trúc Dự Án (Project Structure)
```text
skin-lesion-classification/
├── data/                       # Dữ liệu ảnh HAM10000 và file metadata
│   ├── HAM10000_metadata.csv   # Thông tin lâm sàng (lesion_id, dx, age, sex,...)
│   ├── splits.csv              # File chia tập chuẩn (Zero-Leakage Grouped Split)
│   └── images/                 # 10,015 ảnh dermatoscopic JPG
├── models/
│   ├── __init__.py
│   └── resnet.py               # Kiến trúc mạng ResNet-50 phân loại 7 lớp
├── results/                    # Checkpoint trọng số mô hình và báo cáo
│   ├── baseline/best_model.pth
│   ├── weighted_sampling/best_model.pth (Best Model Checkpoint)
│   ├── focal_loss/best_model.pth
│   ├── augmentation/best_model.pth
│   ├── combined/best_model.pth
│   ├── comparison.csv          # Bảng tổng hợp so sánh 5 cấu hình
│   ├── comparison_chart.png    # Biểu đồ cột so sánh trực quan các chỉ số
│   ├── per_class_recall_comparison.png # So sánh độ nhạy trên các lớp hiếm
│   ├── confusion_matrix_test.png       # Ma trận nhầm lẫn trên tập Test
│   ├── report.pdf              # Báo cáo học thuật chi tiết 11 trang
│   └── ppt.pptx                # Slide thuyết trình bảo vệ 4 trang chuẩn cấu trúc
├── data.py                     # Quản lý Dataset, Transforms, Grouped K-Fold split
├── losses.py                   # Triển khai Cross-Entropy, Focal Loss, Class-Balanced Loss
├── train.py                    # Pipeline huấn luyện, scheduler, checkpointing
├── test.py                     # Script kiểm thử chính thức trên 1,431 ảnh tập Test
├── predict.py                  # Script dự đoán trực tiếp 1 ảnh không cần web server
├── main.py                     # Điểm thực thi chính (hỗ trợ dự đoán trực tiếp & kiểm thử)
├── experiments.py              # Runner tự động chạy và đánh giá benchmark
├── DATA.md                     # Tài liệu chi tiết về đặc tả dữ liệu và quy trình chuẩn bị
├── requirements.txt            # Danh sách thư viện phụ thuộc
└── README.md                   # Hướng dẫn sử dụng và báo cáo tổng quan
```

---

## ⚙️ Cài Đặt Môi Trường (Installation)

```bash
# 1. Kích hoạt môi trường Python (Python 3.10+)
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\activate

# 2. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

---

## 🚀 Hướng Dẫn Sử Dụng (Quick Start)

### 1. Kiểm thử mô hình trên toàn bộ tập Test (Chuẩn đề bài)
Chạy kiểm định mô hình tốt nhất (`Weighted Sampling`) trên toàn bộ 1,431 ảnh kiểm thử độc lập (không rò rỉ dữ liệu). Tự động in bảng chỉ số tổng quan, bảng chi tiết từng lớp và lưu biểu đồ ma trận nhầm lẫn:
```bash
python test.py
```
*(Chỉ định checkpoint tùy chọn nếu muốn so sánh cấu hình khác)*:
```bash
python test.py --checkpoint results/baseline/best_model.pth
```

### 2. Dự đoán trực tiếp ảnh tổn thương da (Không cần Web Server)
Nhập đường dẫn trực tiếp của một file ảnh JPG bất kỳ, script sẽ lập tức tiền xử lý, tính toán phân phối xác suất trên toàn bộ 7 loại bệnh, đưa ra chẩn đoán có xác suất cao nhất cùng mức độ rủi ro lâm sàng:
```bash
# Dự đoán một ảnh cụ thể:
python predict.py --image data/images/ISIC_0024306.jpg

# Hoặc chạy kiểm tra nhanh một ảnh ngẫu nhiên trong dataset:
python predict.py
```

### 3. Thực thi nhanh qua `main.py`
```bash
# Dự đoán ảnh:
python main.py --image data/images/ISIC_0026273.jpg

# Hoặc chạy kiểm thử tập test:
python main.py --test
```

---

## 📈 Kết Quả Thực Nghiệm Thực Tế (Benchmark Results)

Được trích xuất trực tiếp từ các file checkpoint đã huấn luyện thực tế (`results/comparison.csv`):

| Cấu Hình Thực Nghiệm | Loss Function | Chiến Lược Lấy Mẫu | Kỹ Thuật Augmentation | Accuracy | Balanced Acc | Macro F1 (Trọng Tâm) | Weighted F1 | Macro Recall |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | Cross-Entropy | Random | Basic Flip | 83.23% | 58.29% | 0.6197 | 0.8221 | 58.29% |
| **Weighted Sampling** | **Cross-Entropy** | **Weighted Random** | **Basic Flip** | **78.48%** | **68.86%** | **0.6905** | **0.7996** | **68.86%** |
| **Focal Loss** | Focal ($\gamma=2$) | Random | Basic Flip | 79.59% | 65.64% | 0.6550 | 0.8028 | 65.64% |
| **Data Augmentation** | Cross-Entropy | Random | Color + Affine | 81.48% | 57.59% | 0.5922 | 0.8033 | 57.59% |
| **Combined Strategy** | Focal ($\gamma=2$) | Weighted Random | Color + Affine | 68.48% | 67.61% | 0.5845 | 0.7157 | 67.61% |

### Nhận định then chốt từ thực nghiệm:
1. **Bẫy Accuracy thông thường**: Cấu hình Baseline đạt Accuracy cao (83.23%) nhưng chủ yếu do học vẹt lớp chiếm đa số (`nv` chiếm 67% dữ liệu). Balanced Accuracy chỉ đạt 58.29%, nhiều ca ung thư ác tính (`mel`, `bcc`) bị bỏ sót.
2. **Hiệu quả vượt bậc của Weighted Sampling**: Đạt **Macro F1 cao nhất (0.6905)** và **Balanced Accuracy cao nhất (68.86%)**, cải thiện độ nhạy phát hiện ung thư tế bào đáy (`bcc`) lên **79.73%** và ung thư hắc tố (`mel`) lên **66.67%**.

---

## 📄 Báo Cáo & Slide Bảo Vệ
- **Báo cáo chuyên khảo PDF**: `results/report.pdf` (11 trang, đầy đủ công thức toán học, bảng so sánh và phân tích lâm sàng).
- **Slide thuyết trình PPTX**: `results/ppt.pptx` (4 slides chuẩn theo cấu trúc bảo vệ môn học).
