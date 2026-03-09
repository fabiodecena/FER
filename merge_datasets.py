"""
merge_datasets.py – Combina KDEF e MMA con bilanciamento

Strategia:
  - Cap classi grandi MMA a MAX_PER_CLASS
  - Oversample KDEF x KDEF_REPEAT
  - Shuffle e copia in data_merged/
"""

import shutil
import random
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────
KDEF_ROOT = Path("KDEF/data_split")          # KDEF splittato
MMA_ROOT  = Path("MMA/data_mma")       # MMA scaricato
DST = Path("Merged/data_merged")

MAX_PER_CLASS = 6000        # cap per classe per split train (MMA)
MAX_PER_CLASS_VAL = 1000    # cap per classe per val/test
KDEF_REPEAT = 3             # quante volte ripetere KDEF
SEED = 42

SPLITS_MAP = {
    "train": "train",
    "validation": "validation",
    "test": "test",
}

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(root: Path, split: str, cls: str) -> list[Path]:
    """Raccoglie tutte le immagini per split/classe."""
    d = root / split / cls
    if not d.exists():
        return []
    return [f for f in d.iterdir() if f.suffix.lower() in IMG_EXT]


def main():
    random.seed(SEED)

    if DST.exists():
        shutil.rmtree(DST)

    for split in SPLITS_MAP:
        is_train = (split == "train")
        cap = MAX_PER_CLASS if is_train else MAX_PER_CLASS_VAL

        print(f"\n{'='*60}")
        print(f"Split: {split} (cap={cap}, KDEF repeat={KDEF_REPEAT if is_train else 1})")
        print(f"{'='*60}")

        for cls in CLASSES:
            # ── Raccogli immagini ─────────────────────────────────
            kdef_imgs = collect_images(KDEF_ROOT, split, cls)
            mma_imgs  = collect_images(MMA_ROOT, split, cls)

            # ── Oversample KDEF (solo train) ──────────────────────
            repeat = KDEF_REPEAT if is_train else 1
            kdef_expanded = kdef_imgs * repeat

            # ── Cap MMA se necessario ─────────────────────────────
            remaining_slots = max(0, cap - len(kdef_expanded))
            if len(mma_imgs) > remaining_slots:
                random.shuffle(mma_imgs)
                mma_imgs = mma_imgs[:remaining_slots]

            # ── Copia ────────────────────────────────────────────
            dst_dir = DST / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)

            copied = 0

            # Copia KDEF (con prefisso per evitare collisioni nomi)
            for i, img in enumerate(kdef_expanded):
                rep_idx = i // max(len(kdef_imgs), 1)
                dst_name = f"kdef_r{rep_idx}_{img.name}"
                shutil.copy2(img, dst_dir / dst_name)
                copied += 1

            # Copia MMA
            for img in mma_imgs:
                dst_name = f"mma_{img.name}"
                shutil.copy2(img, dst_dir / dst_name)
                copied += 1

            kdef_count = len(kdef_expanded)
            mma_count = len(mma_imgs)
            print(f"  {cls:12s}: KDEF={kdef_count:5d} + MMA={mma_count:5d} = {copied:5d}")

    # ── Riepilogo finale ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Final counts:")
    print(f"{'='*60}")
    for split in SPLITS_MAP:
        total = 0
        for cls in CLASSES:
            d = DST / split / cls
            if d.exists():
                n = len([f for f in d.iterdir() if f.suffix.lower() in IMG_EXT])
                total += n
        print(f"  {split:12s}: {total:6d} images")

    print(f"\n✅ Merged dataset → {DST}")


if __name__ == "__main__":
    main()