"""
dataset.py – Data loading utilities for the FER project.

    data/
    ├── train/
    │   ├── Anger/
    │   ├── Disgust/
    │   ├── Fear/
    │   ├── Happiness/
    │   ├── Neutral/
    │   ├── Sadness/
    │   └── Surprise/
    ├── validation/
    │   └── …  (same sub-folders)
    └── test/
        └── …  (same sub-folders)
"""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# ── Image parameters (EfficientNet-B2 native res = 260) ───────────
IMG_SIZE = 260
BATCH_SIZE = 32
NUM_WORKERS = 8

# ── Transforms (★ stronger augmentation to fight overfitting) ──────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),  # cutout-style
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def get_dataloader(
    root: str | Path,
    split: str = "train",
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, ImageFolder]:
    """
    Return a DataLoader and the underlying ImageFolder my_dataset.
    """
    root = Path(root) / split
    is_train = split == "train"
    my_transforms = train_transforms if is_train else eval_transforms

    my_dataset = ImageFolder(root=root, transform=my_transforms)
    data_loader = DataLoader(
        my_dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader, my_dataset


# ── Quick sanity check ─────────────────────────────────────────────
if __name__ == "__main__":
    loader, dataset = get_dataloader("data", split="train")
    images, labels = next(iter(loader))
    print(f"Batch shape : {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Classes     : {dataset.classes}")