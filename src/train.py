"""
Training script.

Run from inside the `src/` folder, with your virtual environment activated:

    python train.py --data_dir ../data_small --epochs 5 --out ../weights/model.pt

Start with --data_dir pointing at a SMALL subset of your data (e.g. 1000
real + 1000 fake images) to confirm the whole pipeline runs end-to-end before
committing hours to training on the full dataset.
"""

import argparse
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import build_dataloaders
from model import build_model


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)

    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    torch.set_grad_enabled(True)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Folder with train/valid/test subfolders")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="../weights/model.pt", help="Where to save the trained model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected, training will be slow. "
              "If you have an NVIDIA GPU, check that you installed the CUDA "
              "build of PyTorch from pytorch.org.")

    train_loader, valid_loader, test_loader, class_names = build_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )
    print(f"Classes (by index): {class_names}")
    # ImageFolder sorts classes alphabetically, so with folders named
    # 'fake' and 'real' this prints ['fake', 'real'] -> index 0 = fake, 1 = real.
    # model.py's CLASS_NAMES assumes ['real', 'fake']; if your print shows a
    # different order, update model.CLASS_NAMES to match before running inference.

    model = build_model(pretrained=True, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    # Only the unfrozen parameters (the new final layer) get optimized at first.
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, valid_loader, criterion, optimizer, device, train=False)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch}/{args.epochs} ({elapsed:.0f}s) "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            import os
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save(model.state_dict(), args.out)
            print(f"  Saved new best model to {args.out} (val_acc={val_acc:.3f})")

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"Final test accuracy: {test_acc:.3f}")


if __name__ == "__main__":
    main()
