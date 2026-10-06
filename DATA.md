# DATA.md: Skin Lesion Dataset Specification (HAM10000)

## 1. Official Dataset Details & Origin
- **Dataset Name**: HAM10000 ("Human Against Machine with 10,000 training images")
- **Official DOI**: [doi:10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T)
- **Harvard Dataverse URL**: [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- **Kaggle Mirror URL**: [https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Publication Reference**: Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*. Sci Data 5, 180161 (2018).
- **Dataset Version**: Version 2.0 (ISIC Archive 2018 benchmark).
- **Total Samples**: 10,015 dermatoscopic images across 7 diagnostic categories.

---

## 2. Diagnostic Classes & Imbalance Distribution
The HAM10000 dataset exhibits **severe class imbalance**, where the majority class (`nv`) comprises over 66% of the dataset, while the rarest minority class (`df`) comprises barely 1.15%:

| Code | Diagnostic Category | Clinical Description | Sample Count | Percentage | Imbalance Ratio (vs NV) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **nv** | Melanocytic nevi | Benign melanocytic lesions / moles | 6,705 | 66.95% | 1.00 : 1 |
| **mel** | Melanoma | Malignant melanoma | 1,113 | 11.11% | 6.02 : 1 |
| **bkl** | Benign keratosis-like lesions | Seborrheic keratoses, solar lentigines | 1,099 | 10.97% | 6.10 : 1 |
| **bcc** | Basal cell carcinoma | Malignant basal cell carcinoma | 514 | 5.13% | 13.05 : 1 |
| **akiec**| Actinic keratoses / IEC | Actinic keratoses / intraepithelial carcinoma | 327 | 3.27% | 20.50 : 1 |
| **vasc** | Vascular lesions | Angiomas, pyogenic granulomas, hemorrhage | 142 | 1.42% | 47.22 : 1 |
| **df** | Dermatofibroma | Benign dermatofibroma | 115 | 1.15% | 58.30 : 1 |
| **Total**| **7 Classes** | — | **10,015** | **100.0%** | — |

---

## 3. Data Splitting Procedure & Zero Data Leakage Protocol
In medical dermatoscopic imaging, multiple images can be captured from the same physical lesion or lesion site across follow-ups or different angles (indicated by the `lesion_id` column in `HAM10000_metadata.csv`). 

> [!CAUTION]
> **Data Leakage Prevention**: A naive random image-level split will place images of the same patient lesion into both train and test sets, leading to severe test set leakage and deceptively inflated accuracy.

### Leakage-Free Stratified Group Splitting:
We utilize `StratifiedGroupKFold` on the `lesion_id` group identifier while stratifying by class `dx`:
- **Train Set**: 70% (~7,010 images)
- **Validation Set**: 15% (~1,502 images)
- **Test Set**: 15% (~1,503 images)
- **Leakage Verification**:
  $$\text{LesionID}(\text{Train}) \cap \text{LesionID}(\text{Val}) = \emptyset$$
  $$\text{LesionID}(\text{Train}) \cap \text{LesionID}(\text{Test}) = \emptyset$$
  $$\text{LesionID}(\text{Val}) \cap \text{LesionID}(\text{Test}) = \emptyset$$
The fixed splits are persisted deterministically to `data/splits.csv` (Seed = 42).

---

## 4. Preprocessing & Augmentation Pipelines

### Normalization
All input images are converted to RGB, resized to $224 \times 224$ pixels, and normalized using standard ImageNet parameters:
- **Mean**: `[0.485, 0.456, 0.406]`
- **Standard Deviation**: `[0.229, 0.224, 0.225]`

### Augmentation Modes
1. **Basic Augmentation (Baseline)**:
   - Random Horizontal Flip ($p=0.5$)
   - Random Vertical Flip ($p=0.5$)
2. **Advanced Augmentation (Minority Enhancement)**:
   - Random Horizontal Flip ($p=0.5$)
   - Random Vertical Flip ($p=0.5$)
   - Random Rotation ($\pm 20^\circ$)
   - Color Jitter (Brightness $\pm 15\%$, Contrast $\pm 15\%$, Saturation $\pm 15\%$, Hue $\pm 5\%$)
   - Random Affine Translation ($\pm 5\%$)

---

## 5. Script to Reproduce Data
To prepare and verify the leakage-free splits:
```bash
# Generate and verify leakage-free StratifiedGroupKFold splits
python -c "from data import get_dataloaders; get_dataloaders()"
```
