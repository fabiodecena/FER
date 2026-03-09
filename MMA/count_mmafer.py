from pathlib import Path

SRC = Path("data_mma")

for split in ["train", "validation", "test"]:
    split_dir = SRC / split
    if not split_dir.exists():
        print(f"  {split}: NOT FOUND")
        continue
    total = 0
    print(f"\n{split}:")
    for cls_dir in sorted(split_dir.iterdir()):
        if cls_dir.is_dir():
            n = len([f for f in cls_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
            print(f"  {cls_dir.name:12s}: {n:5d}")
            total += n
    print(f"  {'TOTAL':12s}: {total:5d}")