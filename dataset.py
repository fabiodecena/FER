"""
    dataset.py

    DataLoader utilities for MMA FER and similar FER datasets in ImageFolder structure.

    Expected directory structure:
        <root>/
            train/<emotion>/*.jpg
            validation/<emotion>/*.jpg
            test/<emotion>/*.jpg

    Features:
        - Supports split loading (train, validation, test)
        - Multiple configurable augmentation pipelines for training
        - Consistent normalization (ImageNet mean/std)
        - Builds PyTorch DataLoader and torchvision ImageFolder

    Example usage:
        loader, dataset = get_dataloader("Merged/data_merged", split="train", batch_size=64, img_size=96)
"""

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

BATCH_SIZE = 64
NUM_WORKERS = 24
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def build_train_transforms(
    img_size: int = 96,
    augmentation: str = "light",
) -> transforms.Compose:
    """
    Builds a torchvision transform pipeline for training images,
    with configurable augmentation level.

    Args:
        img_size (int): Size to resize/crop images to.
        augmentation (str): One of 'none', 'light', 'medium', or 'heavy'.

    Returns:
        transforms.Compose: Composed transform pipeline.

    Raises:
        ValueError: If an unsupported augmentation level is requested.
    """
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
            transforms.Resize(img_size + 32),  # Resize slightly larger for better cropping
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            # Simulate various distortions and noise
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
            ], p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    raise ValueError(
        f"Unsupported augmentation level: {augmentation!r}. "
        f"Choose from: 'none', 'light', 'medium', 'heavy'."
    )

def build_eval_transforms(img_size: int = 96) -> transforms.Compose:
    """
    Builds a torchvision transform pipeline for evaluation images.

    Args:
        img_size (int): Size to resize images to.

    Returns:
        transforms.Compose: Composed transform pipeline (resize, tensor, normalize).
    """
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
    """
    Builds a PyTorch DataLoader and ImageFolder for the specified split,
    using the appropriate preprocessing pipeline.

    Args:
        root (str or Path): Root directory of dataset.
        split (str): Which split ('train', 'validation', 'test') to load.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.
        augmentation (str): Augmentation level for training ('none', 'light', 'medium', 'heavy').
        img_size (int): Target image size.

    Returns:
        tuple: (DataLoader, ImageFolder dataset)

    Raises:
        FileNotFoundError: If split directory does not exist.
    """
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