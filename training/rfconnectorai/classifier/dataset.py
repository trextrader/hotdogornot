"""
Folder-structured image dataset for the connector classifier.

Expects a directory layout like:
    data_root/
      SMA-M/
        img_0001.jpg
        ...
      SMA-F/
        ...

Folder names become class labels. Subclassing torchvision.datasets.ImageFolder
gives us the right semantics with no extra code, but we centralize it here
so the train + predict modules agree on transforms, ignore non-image files,
and use a stable class-name → index mapping persisted alongside the weights.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 384 was tried (commit 13023b0) and regressed: model collapsed to
# constant 3.5mm-M predictions at 6-class/balanced=289 setup. Reverted
# to 224 — ResNet-18's receptive field doesn't usefully scale to 384
# on our small dataset. Higher resolution only helps when the discriminating
# features are sub-pixel at lower res, which doesn't apply here once
# Hough has already produced a tight crop centered on the connector.
INPUT_SIZE = 224


def make_train_transforms(input_size: int = INPUT_SIZE) -> transforms.Compose:
    resize_before = int(input_size * 1.14)
    return transforms.Compose([
        transforms.Resize(resize_before),
        transforms.RandomResizedCrop(input_size, scale=(0.55, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.15, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.4),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.05), ratio=(0.5, 2.0)),
    ])


def make_eval_transforms(input_size: int = INPUT_SIZE) -> transforms.Compose:
    resize_before = int(input_size * 1.14)
    return transforms.Compose([
        transforms.Resize(resize_before),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ConnectorFolderDataset(Dataset):
    """
    Walks `root/<class_name>/*.{jpg,png,...}` and yields (tensor, class_index)
    pairs. We don't use torchvision.ImageFolder directly because it raises on
    empty subdirs and includes non-image files; this custom class is more
    forgiving for in-progress data folders.
    """

    def __init__(self, root: Path, class_names: list[str], transform=None):
        self.root = Path(root)
        self.class_names = list(class_names)
        self.class_to_idx = {n: i for i, n in enumerate(self.class_names)}
        self.transform = transform

        self.samples: list[tuple[Path, int]] = []
        for cls in self.class_names:
            cls_dir = self.root / cls
            if not cls_dir.is_dir():
                continue
            for p in sorted(cls_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in VALID_EXTS:
                    self.samples.append((p, self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def class_counts(self) -> dict[str, int]:
        counts = {n: 0 for n in self.class_names}
        for _, label in self.samples:
            counts[self.class_names[label]] += 1
        return counts
