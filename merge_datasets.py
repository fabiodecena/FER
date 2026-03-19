import shutil
import random
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# ── Config ────────────────────────────────────────────────────────
KDEF_ROOT = Path("KDEF/data_split")
MMA_ROOT = Path("MMA/data_mma")
DST = Path("Merged/data_merged")

# TARGET BALANCE: KDEF should be roughly 35% of the total per class
TARGET_KDEF_RATIO = 0.35
MAX_MMA_TRAIN = 5000
random.seed(42)

# Mutation Pipeline (Keep this to make KDEF "webcam-ready")
kdef_mutator = T.Compose([
    T.RandomResizedCrop(size=(96, 96), scale=(0.8, 1.0)),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.3, contrast=0.3),
    T.RandomHorizontalFlip(p=0.5),
    T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
])


def merge_balanced():
    if DST.exists(): shutil.rmtree(DST)
    classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    for split in ["train", "validation", "test"]:
        print(f"\n▶ {split.upper()} Distribution:")
        print(f"{'Class':<12} | {'KDEF (Aug)':<10} | {'MMA':<10} | {'Ratio'}")
        print("-" * 55)

        for cls in classes:
            dst_dir = DST / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)

            # 1. Count available images
            k_src = KDEF_ROOT / split / cls
            m_src = MMA_ROOT / split / cls
            k_orig_imgs = list(k_src.glob("*")) if k_src.exists() else []
            m_orig_imgs = list(m_src.glob("*")) if m_src.exists() else []

            # 2. Cap MMA
            cap = MAX_MMA_TRAIN if split == "train" else 500
            random.shuffle(m_orig_imgs)
            selected_mma = m_orig_imgs[:cap]
            m_count = len(selected_mma)

            # 3. Calculate dynamic KDEF variants for TRAIN
            if split == "train" and len(k_orig_imgs) > 0 and m_count > 0:
                # Formula: How many KDEF total do we need to hit 35%?
                target_total = m_count / (1 - TARGET_KDEF_RATIO)
                target_k_total = target_total - m_count
                variants_per_img = max(1, int(target_k_total / len(k_orig_imgs)))
            else:
                variants_per_img = 1

            # 4. Save KDEF
            k_final_count = 0
            for img_path in k_orig_imgs:
                with Image.open(img_path).convert("RGB") as img:
                    if split == "train":
                        for i in range(variants_per_img):
                            kdef_mutator(img).save(dst_dir / f"k_v{i}_{img_path.name}")
                            k_final_count += 1
                    else:
                        img.save(dst_dir / f"k_orig_{img_path.name}")
                        k_final_count += 1

            # 5. Save MMA
            for img_path in selected_mma:
                shutil.copy2(img_path, dst_dir / f"m_{img_path.name}")

            ratio = k_final_count / (k_final_count + m_count) if (k_final_count + m_count) > 0 else 0
            print(f"{cls:<12} | {k_final_count:<10} | {m_count:<10} | {ratio:.1%}")


if __name__ == "__main__":
    merge_balanced()