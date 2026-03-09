"""
dataset.py – DataLoader for MMA FER dataset (ImageFolder structure)

Expected structure:
    data_mma/
        train/<emotion>/*.jpg
        validation/<emotion>/*.jpg
        test/<emotion>/*.jpg
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMG_SIZE = 96
BATCH_SIZE = 64
NUM_WORKERS = 8

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transforms(augmentation: str = "light") -> transforms.Compose:
    augmentation = augmentation.lower()

    if augmentation == "none":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    if augmentation == "light":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    if augmentation == "medium":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.20,
                contrast=0.20,
                saturation=0.15,
                hue=0.03,
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
        ])

    raise ValueError(
        f"Unsupported augmentation level: {augmentation!r}. "
        f"Choose from: 'none', 'light', 'medium'."
    )


eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_dataloader(
    root: str | Path,
    split: str = "train",
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    augmentation: str = "light",
) -> tuple[DataLoader, ImageFolder]:
    split_dir = Path(root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    is_train = split == "train"
    tfm = build_train_transforms(augmentation) if is_train else eval_transforms

    dataset = ImageFolder(root=split_dir, transform=tfm)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )
    return loader, dataset