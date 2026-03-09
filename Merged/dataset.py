"""
dataset.py – DataLoader per KDEF (struttura ImageFolder)

Struttura attesa:
    data_split/
        train/<emotion>/*.jpg
        validation/<emotion>/*.jpg
        test/<emotion>/*.jpg
"""

from __future__ import annotations

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 8

# Normalizzazione ImageNet (standard per backbone timm pretrained)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

eval_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_dataloader(
    root: str | Path,
    split: str = "train",
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, ImageFolder]:
    """
    Carica un split come ImageFolder.
    root: cartella base (es. 'data_split')
    split: 'train', 'validation' o 'test'
    """
    split_dir = Path(root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    is_train = (split == "train")
    tfm = train_transforms if is_train else eval_transforms

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