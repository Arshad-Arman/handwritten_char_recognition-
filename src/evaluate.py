"""Evaluation utilities + CLI:  python -m src.evaluate --model models/cnn_mnist.pt"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    """Return (avg_loss, accuracy, predictions, targets)."""
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if criterion is not None:
            total_loss += criterion(out, y).item() * len(y)
        preds.append(out.argmax(1).cpu())
        targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return total_loss / len(targets), float((preds == targets).mean()), preds, targets


def confusion_matrix(preds, targets, k):
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (targets, preds), 1)
    return cm


def plot_confusion(cm, classes, path):
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    size = max(6, len(classes) * 0.28)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), list(classes), fontsize=7)
    ax.set_yticks(range(len(classes)), list(classes), fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalised confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(hist, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(hist["train_loss"], label="train")
    axes[0].plot(hist["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(hist["train_acc"], label="train")
    axes[1].plot(hist["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_class_report(cm, classes, worst=5):
    acc = np.diag(cm) / np.maximum(cm.sum(1), 1)
    order = np.argsort(acc)[:worst]
    return [(classes[i], float(acc[i])) for i in order]


def main(argv=None):
    from .data import get_loaders
    from .model import CNN

    ap = argparse.ArgumentParser(description="Evaluate a trained CNN on its test set")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--synthetic-size", type=int, default=20000)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.model, map_location=device)
    if ck.get("arch") != "cnn":
        raise SystemExit("evaluate.py only handles CNN checkpoints")
    model = CNN(len(ck["classes"])).to(device)
    model.load_state_dict(ck["state_dict"])
    _, _, test_loader, classes = get_loaders(ck["dataset"], args.data_dir, args.batch_size,
                                             augment=False, synthetic_size=args.synthetic_size)
    _, acc, preds, targets = evaluate(model, test_loader, device)
    cm = confusion_matrix(preds, targets, len(classes))
    out = os.path.splitext(args.model)[0] + "_confusion.png"
    plot_confusion(cm, classes, out)
    print(f"test accuracy: {acc * 100:.2f}%   (confusion matrix -> {out})")
    print("hardest classes:", per_class_report(cm, classes))


if __name__ == "__main__":
    main()
