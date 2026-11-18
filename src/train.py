"""Train a binary crack classifier using ImageFolder directories.

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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/concrete")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    seed_everything(args.seed)

    root = Path(args.data_dir)
    if not root.exists():
        raise SystemExit(f"Dataset directory not found: {root}")
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root, transform=transform)
    if len(dataset.classes) != 2:
        raise SystemExit(f"Expected exactly 2 class folders; found {dataset.classes}")
    indices = np.arange(len(dataset))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=args.seed, stratify=dataset.targets
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=args.seed,
        stratify=np.array(dataset.targets)[temp_idx]
    )
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val = float("inf")
    output = Path("outputs")
    output.mkdir(exist_ok=True)

    history = []
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        model.eval()
        val_loss, correct, count = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in eval_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss += criterion(logits, labels).item() * images.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                count += labels.size(0)
        train_loss = running_loss / len(train_idx)
        val_loss /= max(count, 1)
        val_acc = correct / max(count, 1)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc})
        print(f"Epoch {epoch+1}/{args.epochs} - train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_accuracy={val_acc:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "classes": dataset.classes,
                        "image_size": args.image_size, "seed": args.seed}, output / "best_model.pth")

    (output / "training_history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved best checkpoint to {output / 'best_model.pth'}")
    print(f"Class mapping: {dict(enumerate(dataset.classes))}")
    print(f"Split sizes: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")
    print("Note: run evaluate.py to obtain held-out test metrics.")


if __name__ == "__main__":
    main()
