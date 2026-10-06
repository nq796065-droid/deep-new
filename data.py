"""
data.py - Data Processing and Loading Module for HAM10000.
Key scientific design principles:
  1. Data Splitting: Lesion-level Stratified Group Split on `lesion_id`.
     Scientific note: HAM10000 provides `lesion_id` without patient_id.
     Grouping on `lesion_id` guarantees zero morphological leakage of identical lesions
     across Train, Val, and Test partitions.
  2. Exception Handling: Explicitly raises RuntimeError if evaluation images fail to load
     to preserve metric integrity; logs warning on training split.
  3. Optimized Augmentation: Merges rotation and affine transformations into a single matrix step
     to prevent image blurring from double interpolation.
  4. Full Reproducibility: Enforces cudnn.deterministic=True, cudnn.benchmark=False,
     combined with seed_worker and explicit PyTorch Generator.
"""

from pathlib import Path
import random
import logging
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold


logger = logging.getLogger(__name__)

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

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


def set_seed(seed: int = 42):
    """Set synchronized random seed across Python, NumPy, PyTorch, and CUDA CuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Worker initialization function for reproducible multi-process DataLoader."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_transforms(is_train: bool = True, use_aug: bool = False):
    """
    Image transformation pipeline.
    - Advanced Training (+use_aug): Combines rotation and affine into a single transformation
      to avoid double interpolation blur.
    """
    if is_train:
        t_list = [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]
        if use_aug:
            t_list.extend([
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
                transforms.RandomAffine(degrees=25, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ])
        t_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
        return transforms.Compose(t_list)

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset loading images directly from path and metadata.
    Metric safety: Raises RuntimeError if evaluation images fail to load.
    """
    def __init__(self, df: pd.DataFrame, img_dir: Path, transform=None, is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_train = is_train
        self.labels = [CLASS_TO_IDX[dx] for dx in self.df['dx']]
        self.image_paths = [str(self.img_dir / f"{iid}.jpg") for iid in self.df['image_id']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        try:
            with Image.open(path) as raw_img:
                img = raw_img.convert('RGB')
        except Exception as exc:
            if not self.is_train:
                raise RuntimeError(
                    f"Failed to read evaluation image at '{path}': {exc}. "
                    "Cannot continue evaluation to avoid metric distortion."
                )
            logger.warning("Failed to read training image '%s': %s. Replacing with zero image.", path, exc)
            img = Image.new('RGB', (224, 224))

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]


def get_dataloaders(
    data_dir: str = "data",
    batch_size: int = 32,
    use_weighted_sampler: bool = False,
    use_aug: bool = False,
    num_workers: int = 0,
    pin_memory: bool = None
):
    """
    Initialize Train, Val, Test DataLoaders with zero-leakage Lesion-Level Grouped Split.
    """
    data_path = Path(data_dir)
    splits_file = data_path / "splits.csv"

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if not splits_file.exists():
        set_seed(42)
        meta = pd.read_csv(data_path / "HAM10000_metadata.csv")
        # Tier 1: Split Train (80%) vs Temp (20%) on lesion_id
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, temp_idx = next(sgkf.split(meta, meta['dx'], meta['lesion_id']))
        temp_meta = meta.iloc[temp_idx]

        # Tier 2: Split Temp into Val (10%) and Test (10%) on lesion_id
        sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
        v_local, t_local = next(sgkf2.split(temp_meta, temp_meta['dx'], temp_meta['lesion_id']))

        meta['split'] = 'train'
        meta.loc[temp_meta.iloc[v_local].index, 'split'] = 'val'
        meta.loc[temp_meta.iloc[t_local].index, 'split'] = 'test'
        meta.to_csv(splits_file, index=False)

    df = pd.read_csv(splits_file)
    img_dir = data_path / "images"

    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']

    train_ds = HAM10000Dataset(train_df, img_dir, get_transforms(is_train=True, use_aug=use_aug), is_train=True)
    val_ds = HAM10000Dataset(val_df, img_dir, get_transforms(is_train=False), is_train=False)
    test_ds = HAM10000Dataset(test_df, img_dir, get_transforms(is_train=False), is_train=False)

    g = torch.Generator()
    g.manual_seed(42)

    sampler = None
    if use_weighted_sampler:
        class_counts = train_df['dx'].value_counts()
        sample_weights = [1.0 / class_counts[dx] for dx in train_df['dx']]
        tensor_weights = torch.as_tensor(sample_weights, dtype=torch.double)
        sampler = WeightedRandomSampler(
            tensor_weights,
            num_samples=len(tensor_weights),
            replacement=True,
            generator=g
        )

    persistent = (num_workers > 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=persistent
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent
    )

    return train_loader, val_loader, test_loader
