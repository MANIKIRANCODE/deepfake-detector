"""
Builds a small subset of the full dataset for a fast first training run.

Copies the first N files from each data/<split>/<class> folder into the
matching data_small/<split>/<class> folder.

Usage (run from inside the scripts folder):
    python make_small_subset.py

Edit TRAIN_COUNT / VALID_COUNT / TEST_COUNT below if you want different sizes.
"""

import os
import shutil

SOURCE_DIR = os.path.join("..", "data")
DEST_DIR = os.path.join("..", "data_small")

# How many images per class to copy for each split.
COUNTS = {
    "train": 1000,
    "valid": 200,
    "test": 200,
}

CLASSES = ["real", "fake"]


def main():
    for split, count in COUNTS.items():
        for class_name in CLASSES:
            src_folder = os.path.join(SOURCE_DIR, split, class_name)
            dst_folder = os.path.join(DEST_DIR, split, class_name)

            if not os.path.isdir(src_folder):
                print(f"SKIP: {src_folder} does not exist.")
                continue

            os.makedirs(dst_folder, exist_ok=True)

            files = sorted(os.listdir(src_folder))[:count]
            for fname in files:
                src_path = os.path.join(src_folder, fname)
                dst_path = os.path.join(dst_folder, fname)
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)

            print(f"Copied {len(files)} files: {src_folder} -> {dst_folder}")

    print("Done. data_small is ready.")


if __name__ == "__main__":
    main()
