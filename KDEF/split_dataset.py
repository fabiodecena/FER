"""
Crea split train/validation/test identity-disjoint.

Naming: {subject}_{progressive}.jpg
Struttura input:  data/<emotion>/*.jpg
Struttura output: data_split/train/<emotion>/*.jpg
                  data_split/validation/<emotion>/*.jpg
                  data_split/test/<emotion>/*.jpg
"""

import shutil
import random
from pathlib import Path
from collections import defaultdict

SRC = Path("data")
DST = Path("data_split")

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

SEED = 42


def extract_subject_id(filename: str) -> str:
    return filename.split("_")[0]


def main():
    random.seed(SEED)

    # ── 1. Raccogli immagini per soggetto ─────────────────────────
    subject_images: dict[str, list[tuple[str, Path]]] = defaultdict(list)

    classes = sorted([d.name for d in SRC.iterdir() if d.is_dir()])
    print(f"Classes: {classes}")

    for cls_dir in SRC.iterdir():
        if not cls_dir.is_dir():
            continue
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                subj = extract_subject_id(img_path.name)
                subject_images[subj].append((cls_dir.name, img_path))

    subjects = sorted(subject_images.keys(), key=int)
    print(f"Total subjects: {len(subjects)}")
    print(f"Total images:   {sum(len(v) for v in subject_images.values())}")

    # ── 2. Shuffle e split per soggetto ───────────────────────────
    random.shuffle(subjects)

    n = len(subjects)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_subjects = subjects[:n_train]
    val_subjects   = subjects[n_train:n_train + n_val]
    test_subjects  = subjects[n_train + n_val:]

    print(f"\nTrain subjects: {len(train_subjects)}")
    print(f"Val subjects:   {len(val_subjects)}")
    print(f"Test subjects:  {len(test_subjects)}")

    # ── 3. Copia ──────────────────────────────────────────────────
    if DST.exists():
        shutil.rmtree(DST)

    splits = {
        "train":      train_subjects,
        "validation": val_subjects,
        "test":       test_subjects,
    }

    for split_name, split_subjects in splits.items():
        count = 0
        class_count = defaultdict(int)
        for subj in split_subjects:
            for cls_name, img_path in subject_images[subj]:
                dst_dir = DST / split_name / cls_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dst_dir / img_path.name)
                count += 1
                class_count[cls_name] += 1
        print(f"\n  {split_name}: {count} images")
        for cls in sorted(class_count):
            print(f"    {cls:12s}: {class_count[cls]}")


if __name__ == "__main__":
    main()