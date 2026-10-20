"""Dataset loading: MNIST, EMNIST (all splits) and a synthetic fallback.

Everything is turned into plain numpy arrays first (``load_raw``) so the same
upright 28x28 images feed both the CNN classifier and the CRNN word generator.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .constants import MEAN, STD
from .synthetic import DIGITS, UPPER, generate

LOWER = "abcdefghijklmnopqrstuvwxyz"

CLASS_MAPS = {
    "mnist": DIGITS,                                              # 10 classes
    "emnist-digits": DIGITS,                                      # 10 classes
    "emnist-letters": UPPER,                                      # 26 classes (case merged)
    "emnist-balanced": DIGITS + UPPER + "abdefghnqrt",            # 47 classes
    "emnist-byclass": DIGITS + UPPER + LOWER,                     # 62 classes
    "synthetic": DIGITS + UPPER,                                  # 36 classes, fonts (no download)
}


def load_raw(name: str, root: str = "data", train: bool = True, synthetic_size: int = 20000):
    """Return (images uint8 [N,28,28] upright, labels int64 [N], classes str)."""
    if name not in CLASS_MAPS:
        raise ValueError(f"unknown dataset '{name}', choose from {list(CLASS_MAPS)}")
    classes = CLASS_MAPS[name]
    if name == "synthetic":
        imgs, labels = generate(synthetic_size, classes, seed=1 if train else 2)
        return imgs, labels, classes
    if name == "mnist":
        ds = datasets.MNIST(root, train=train, download=True)
        imgs, labels = ds.data.numpy(), ds.targets.numpy()
    else:
        split = name.split("-", 1)[1]
        ds = datasets.EMNIST(root, split=split, train=train, download=True)
        # EMNIST images are stored transposed -> transpose to make them upright.
        imgs, labels = ds.data.numpy().transpose(0, 2, 1), ds.targets.numpy()
        if split == "letters":  # raw labels are 1..26
            labels = labels - 1
    return np.ascontiguousarray(imgs), labels.astype(np.int64), classes


class ArrayCharDataset(Dataset):
    """In-memory (image, label) dataset with optional light augmentation."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.images, self.labels = images, labels
        self.aug = (transforms.RandomAffine(degrees=10, translate=(0.08, 0.08),
                                            scale=(0.9, 1.1), shear=8) if augment else None)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        if self.aug is not None:
            img = np.asarray(self.aug(Image.fromarray(img)))
        x = (torch.from_numpy(np.array(img, copy=True)).float().unsqueeze(0) / 255.0 - MEAN) / STD
        return x, int(self.labels[i])


def get_loaders(name: str, root: str = "data", batch_size: int = 128, augment: bool = True,
                val_frac: float = 0.1, workers: int = 0, synthetic_size: int = 20000, seed: int = 42):
    imgs, labels, classes = load_raw(name, root, True, synthetic_size)
    perm = np.random.RandomState(seed).permutation(len(labels))
    n_val = int(len(labels) * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    train_full = ArrayCharDataset(imgs, labels, augment=augment)
    val_full = ArrayCharDataset(imgs, labels, augment=False)
    test_imgs, test_labels, _ = load_raw(name, root, False, max(synthetic_size // 5, 2000))
    test_ds = ArrayCharDataset(test_imgs, test_labels)
    kw = dict(batch_size=batch_size, num_workers=workers)
    return (DataLoader(Subset(train_full, tr_idx), shuffle=True, **kw),
            DataLoader(Subset(val_full, val_idx), shuffle=False, **kw),
            DataLoader(test_ds, shuffle=False, **kw),
            classes)
