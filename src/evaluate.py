"""Evaluate a trained checkpoint on the same deterministic held-out test split."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/concrete")
    parser.add_argument("--checkpoint", default="outputs/best_model.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    seed = int(checkpoint["seed"])
    image_size = int(checkpoint["image_size"])
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    if dataset.classes != checkpoint["classes"]:
        raise SystemExit(
            f"Dataset classes {dataset.classes} do not match checkpoint classes "
            f"{checkpoint['classes']}"
        )

    targets = np.asarray(dataset.targets)
    indices = np.arange(len(dataset))
    _, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=seed, stratify=targets
    )
    _, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed, stratify=targets[temp_idx]
    )
    loader = DataLoader(
        Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
            y_score.extend(probabilities[:, 1].tolist())
            y_pred.extend(probabilities.argmax(axis=1).tolist())
            y_true.extend(labels.numpy().tolist())

    label_ids = list(range(len(dataset.classes)))
    result = {
        "test_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        "macro_recall": float(recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) == 2 else None,
        "class_names": dataset.classes,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=label_ids
        ).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=label_ids, target_names=dataset.classes,
            zero_division=0, output_dict=True
        ),
    }
    print(json.dumps(result, indent=2))
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/test_metrics.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
