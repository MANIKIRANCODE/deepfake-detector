"""
Optional helper: only needed if your raw images are full photos (not already
cropped to faces). The Kaggle "140k Real and Fake Faces" dataset is already
cropped, so you likely don't need this script with that dataset.

Usage:
    python prepare_dataset.py --in_dir raw_photos/real --out_dir data/train/real
    python prepare_dataset.py --in_dir raw_photos/fake --out_dir data/train/fake

Run it once per class per split (train/valid/test) you need to build.
"""

import argparse
import os
import sys

import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from face_utils import FaceCropper

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", required=True, help="Folder of raw photos")
    parser.add_argument("--out_dir", required=True, help="Folder to write cropped faces to")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cropper = FaceCropper()

    files = [f for f in os.listdir(args.in_dir) if os.path.splitext(f)[1].lower() in IMG_EXTS]
    skipped = 0

    for i, fname in enumerate(files, 1):
        path = os.path.join(args.in_dir, fname)
        bgr = cv2.imread(path)
        if bgr is None:
            skipped += 1
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        crop = cropper.crop_largest_face(rgb)
        if crop is None:
            skipped += 1
            continue
        out_path = os.path.join(args.out_dir, fname)
        cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        if i % 100 == 0:
            print(f"Processed {i}/{len(files)}")

    print(f"Done. Wrote {len(files) - skipped} crops, skipped {skipped} (no face found or unreadable).")


if __name__ == "__main__":
    main()
