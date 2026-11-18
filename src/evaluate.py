"""Evaluate a saved model on a deterministic held-out test split."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/concrete")
    parser.add_argument("--checkpoint", default="outputs/best_model.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    seed = int(checkpoint["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    size = int(checkpoint["image_size"])
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    indices = np.arange(len(dataset))
    _, temp_idx = train_test_split(indices, test_size=0.30, random_state=seed, stratify=dataset.targets)
    _, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed,
                                   stratify=np.array(dataset.targets)[temp_idx])
    loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(1).cpu().numpy()
            y_pred.extend(predictions.tolist())
            y_true.extend(labels.numpy().tolist())

    labels = list(range(len(dataset.classes)))
    result = {
        "test_samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "class_names": dataset.classes,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=labels,
                                                       target_names=dataset.classes, zero_division=0,
                                                       output_dict=True),
    }
    print(json.dumps(result, indent=2))
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/test_metrics.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
