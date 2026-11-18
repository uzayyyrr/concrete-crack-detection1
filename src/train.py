"""Train a binary concrete-crack classifier with leakage-safe transforms.

Expected layout:
data/concrete/Negative/*.jpg
data/concrete/Positive/*.jpg
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/concrete")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive")
    seed_everything(args.seed)

    root = Path(args.data_dir)
    if not root.is_dir():
        raise SystemExit(f"Dataset directory not found: {root}")

    # Read labels once, then create separate datasets so random augmentation
    # is applied to training images only (never validation/test images).
    label_dataset = datasets.ImageFolder(root)
    if len(label_dataset.classes) != 2:
        raise SystemExit(f"Expected exactly 2 class folders; found {label_dataset.classes}")
    targets = np.asarray(label_dataset.targets)
    indices = np.arange(len(label_dataset))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=args.seed, stratify=targets
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=args.seed, stratify=targets[temp_idx]
    )

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    train_data = datasets.ImageFolder(root, transform=train_transform)
    eval_data = datasets.ImageFolder(root, transform=eval_transform)
    if train_data.classes != eval_data.classes:
        raise RuntimeError("Class folders changed while preparing datasets")

    train_loader = DataLoader(
        Subset(train_data, train_idx), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        Subset(eval_data, val_idx), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    output = Path("outputs")
    output.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * images.size(0)

        model.eval()
        total_val_loss, correct, count = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                total_val_loss += criterion(logits, labels).item() * images.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                count += labels.size(0)

        train_loss = total_train_loss / len(train_idx)
        val_loss = total_val_loss / max(count, 1)
        val_accuracy = correct / max(count, 1)
        record = {
            "epoch": epoch + 1, "train_loss": train_loss,
            "validation_loss": val_loss, "validation_accuracy": val_accuracy
        }
        history.append(record)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_accuracy={val_accuracy:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "classes": train_data.classes,
                "image_size": args.image_size,
                "seed": args.seed,
                "split_sizes": {
                    "train": len(train_idx), "validation": len(val_idx),
                    "test": len(test_idx)
                },
                "config": vars(args),
            }, output / "best_model.pth")

    (output / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved best checkpoint: {output / 'best_model.pth'}")
    print(f"Classes: {train_data.classes}")
    print(f"Split sizes: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")
    print("Run src/evaluate.py after training to measure held-out test performance.")


if __name__ == "__main__":
    main()
