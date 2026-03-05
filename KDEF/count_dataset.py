from pathlib import Path
from collections import defaultdict

SRC = Path("data")

subjects = set()
class_counts = defaultdict(int)
subject_counts = defaultdict(set)

for cls_dir in sorted(SRC.iterdir()):
    if not cls_dir.is_dir():
        continue
    for img in cls_dir.iterdir():
        if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
            subj = img.name.split("_")[0]
            subjects.add(subj)
            class_counts[cls_dir.name] += 1
            subject_counts[cls_dir.name].add(subj)

print(f"Total subjects: {len(subjects)}")
print(f"Total images:   {sum(class_counts.values())}")
print()
for cls in sorted(class_counts):
    print(f"  {cls:12s}: {class_counts[cls]:4d} images, {len(subject_counts[cls]):3d} subjects")