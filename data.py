"""
data.py - Xu ly Du lieu, Dataset, Transforms va DataLoaders.
Khong ro ri du lieu (Zero-Leakage Grouped Split theo lesion_id).
Tong cong: ~95 dong code.
"""

from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_FULL_NAMES = {
    'akiec': 'Actinic keratoses / intraepithelial carcinoma',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions'
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(is_train: bool = True, use_aug: bool = False):
    """Pipeline tang cuong du lieu anh."""
    if is_train:
        t_list = [transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        if use_aug:
            t_list.extend([transforms.ColorJitter(brightness=0.15, contrast=0.15), transforms.RandomRotation(20)])
        t_list.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        return transforms.Compose(t_list)
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


class HAM10000Dataset(Dataset):
    """PyTorch Dataset doc anh tu thu muc va nhan tu DataFrame."""
    def __init__(self, df: pd.DataFrame, img_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = [CLASS_NAMES.index(dx) for dx in self.df['dx']]
        self.image_paths = [self.img_dir / f"{iid}.jpg" for iid in self.df['image_id']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_dataloaders(data_dir: str = "data", batch_size: int = 32, use_weighted_sampler: bool = False, use_aug: bool = False):
    """Tao Train, Val, Test DataLoaders voi co che chong ro ri du lieu."""
    data_path = Path(data_dir)
    splits_file = data_path / "splits.csv"
    
    if not splits_file.exists():
        # Tao split chong ro ri theo benh nhan (lesion_id) neu chua co
        meta = pd.read_csv(data_path / "HAM10000_metadata.csv")
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, temp_idx = next(sgkf.split(meta, meta['dx'], meta['lesion_id']))
        meta['split'] = 'train'
        val_idx = temp_idx[:len(temp_idx)//2]
        test_idx = temp_idx[len(temp_idx)//2:]
        meta.loc[val_idx, 'split'] = 'val'
        meta.loc[test_idx, 'split'] = 'test'
        meta.to_csv(splits_file, index=False)

    df = pd.read_csv(splits_file)
    img_dir = data_path / "images"

    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']

    train_ds = HAM10000Dataset(train_df, img_dir, get_transforms(is_train=True, use_aug=use_aug))
    val_ds = HAM10000Dataset(val_df, img_dir, get_transforms(is_train=False))
    test_ds = HAM10000Dataset(test_df, img_dir, get_transforms(is_train=False))

    sampler = None
    if use_weighted_sampler:
        # Tinh trong so mau theo nghich dao tan suat lop
        class_counts = train_df['dx'].value_counts()
        weights = [1.0 / class_counts[dx] for dx in train_df['dx']]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
