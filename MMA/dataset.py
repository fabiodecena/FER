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

BATCH_SIZE = 64
NUM_WORKERS = 8
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transforms(
    img_size: int = 96,
    augmentation: str = "light",
) -> transforms.Compose:
    augmentation = augmentation.lower()

    if augmentation == "none":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    if augmentation == "light":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
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
            transforms.Resize((img_size, img_size)),
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

    if augmentation == "heavy":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.3,
                hue=0.08,
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        ])

    raise ValueError(
        f"Unsupported augmentation level: {augmentation!r}. "
        f"Choose from: 'none', 'light', 'medium', 'heavy'."
    )


def build_eval_transforms(img_size: int = 96) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_dataloader(
    root: str | Path,
    split: str = "train",
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    augmentation: str = "light",
    img_size: int = 96,
) -> tuple[DataLoader, ImageFolder]:
    split_dir = Path(root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    is_train = split == "train"
    tfm = (
        build_train_transforms(img_size=img_size, augmentation=augmentation)
        if is_train
        else build_eval_transforms(img_size=img_size)
    )

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