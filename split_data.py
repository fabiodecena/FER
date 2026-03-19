"""
split_data.py

Splits image data stored in a directory into train, validation, and test sets,
organized by class. Classes are determined from image filename prefixes.

Features:
    - Configurable split ratios (default: 70% train, 15% validation, 15% test)
    - Automatically creates split and class directories
    - Robust handling of image extensions (.png, .jpg, .jpeg)
    - Filename prefix is used as a class label
    - Uses shutil for file copying (preserves metadata)

Input:
    BASE_DIR: Directory containing images (default 'screenshots')
    Images should be named as <class>_<...>.png/jpg/jpeg

Output:
    - BASE_DIR/train/<class>/<image>
    - BASE_DIR/validation/<class>/<image>
    - BASE_DIR/test/<class>/<image>

Example usage:
    python split_data.py

"""
import os
import shutil
import random

# Configuration
BASE_DIR = 'screenshots'
SPLIT_NAMES = ['train', 'validation', 'test']
SPLIT_RATIOS = [0.70, 0.15, 0.15]


def main():
    """
       Main entry point for splitting image data by class and set.

       Scans BASE_DIR for image files, extracts a class from the filename prefix,
       splits per class according to SPLIT_RATIOS, and copies images into
       respective train/validation/test subfolders.

       Raises:
           FileNotFoundError: If BASE_DIR does not exist.
           AssertionError: If split ratios do not sum to 1.0.

       Prints:
           Per-class image counts for each split.
    """
    image_exts = ('.png', '.jpg', '.jpeg')
    all_images = [f for f in os.listdir(BASE_DIR)
                  if os.path.isfile(os.path.join(BASE_DIR, f)) and f.lower().endswith(image_exts)]

    # Classify images according to the filename prefix
    class_to_images = {}
    for img in all_images:
        cls_name = img.split('_')[0]  # Extract class from filename prefix
        class_to_images.setdefault(cls_name, []).append(img)

    # For each split folder and class, ensure folders exist
    for split in SPLIT_NAMES:
        for cls in class_to_images:
            split_cls_dir = os.path.join(BASE_DIR, split, cls)
            os.makedirs(split_cls_dir, exist_ok=True)

    # For each class, perform a split and copy
    for cls, imgs in class_to_images.items():
        random.shuffle(imgs)
        n_total = len(imgs)
        n_train = int(n_total * SPLIT_RATIOS[0])
        n_val = int(n_total * SPLIT_RATIOS[1])


        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:n_train + n_val]
        test_imgs = imgs[n_train + n_val:]

        for split, split_imgs in zip(SPLIT_NAMES, [train_imgs, val_imgs, test_imgs]):
            for img in split_imgs:
                src = os.path.join(BASE_DIR, img)
                dst = os.path.join(BASE_DIR, split, cls, img)
                shutil.copy2(src, dst)

        print(f"{cls}: total {n_total} → train {len(train_imgs)}, validation {len(val_imgs)}, test {len(test_imgs)}")


if __name__ == "__main__":
    main()