"""
Dataset loading.

Expects your data folder laid out like this (this is exactly how the Kaggle
"140k Real and Fake Faces" dataset is already organized, so if you used that
dataset you don't need to rearrange anything):

    data/
        train/
            real/   *.jpg
            fake/   *.jpg
        valid/
            real/   *.jpg
            fake/   *.jpg
        test/
            real/   *.jpg
            fake/   *.jpg

Each image in these folders should already be a cropped face. If your own
images are full photos (not pre-cropped), run `scripts/prepare_dataset.py`
first to crop faces out of them before training.
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import IMAGE_SIZE

# Standard ImageNet normalization stats -- required because the pretrained
# backbone was trained with images normalized this way.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        # Light augmentation: helps the model generalize instead of
        # memorizing exact pixels. Keep it mild -- faces are sensitive to
        # over-distortion.
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def build_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 2):
    """
    data_dir: path to the folder containing train/, valid/, test/ subfolders.
    Returns (train_loader, valid_loader, test_loader, class_names).
    """
    splits = {}
    for split_name, is_train in [("train", True), ("valid", False), ("test", False)]:
        split_path = os.path.join(data_dir, split_name)
        if not os.path.isdir(split_path):
            raise FileNotFoundError(
                f"Expected a folder at {split_path}. "
                "Check your --data_dir and folder layout against the docstring "
                "at the top of dataset.py."
            )
        dataset = datasets.ImageFolder(split_path, transform=build_transforms(is_train))
        splits[split_name] = dataset

    class_names = splits["train"].classes  # e.g. ['fake', 'real'], alphabetical
    loaders = {
        name: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        for name, ds in splits.items()
    }
    return loaders["train"], loaders["valid"], loaders["test"], class_names
