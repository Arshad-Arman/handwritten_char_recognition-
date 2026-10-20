"""Train the CNN classifier.

Examples
    python -m src.train --dataset mnist --epochs 8
    python -m src.train --dataset emnist-balanced --epochs 15
    python -m src.train --dataset synthetic --epochs 3        # no download, quick test
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .data import CLASS_MAPS, get_loaders
from .evaluate import confusion_matrix, evaluate, per_class_report, plot_confusion, plot_history
from .model import CNN


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train a CNN for handwritten character recognition")
    ap.add_argument("--dataset", default="mnist", choices=list(CLASS_MAPS))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (keep 0 on Windows)")
    ap.add_argument("--synthetic-size", type=int, default=20000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    print(f"device: {device} | dataset: {args.dataset}")

    train_loader, val_loader, test_loader, classes = get_loaders(
        args.dataset, args.data_dir, args.batch_size, augment=not args.no_augment,
        workers=args.workers, synthetic_size=args.synthetic_size, seed=args.seed)
    print(f"classes: {len(classes)} | train batches: {len(train_loader)}")

    model = CNN(len(classes)).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for step, (x, y) in enumerate(train_loader, 1):
            if step % 50 == 0 or step == len(train_loader):
                print(f"  epoch {epoch:02d} batch {step}/{len(train_loader)} "
                      f"({time.time() - t0:.0f}s elapsed)", flush=True)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            seen += len(y)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, criterion)
        for k, v in zip(hist, (loss_sum / seen, correct / seen, val_loss, val_acc)):
            hist[k].append(float(v))
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:02d}/{args.epochs} | train loss {hist['train_loss'][-1]:.4f} "
              f"acc {hist['train_acc'][-1] * 100:.2f}% | val loss {val_loss:.4f} "
              f"acc {val_acc * 100:.2f}% | {time.time() - t0:.0f}s")

    model.load_state_dict(best_state)
    _, test_acc, preds, targets = evaluate(model, test_loader, device)
    print(f"\nBEST val acc {best_acc * 100:.2f}% | TEST acc {test_acc * 100:.2f}%")

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f"cnn_{args.dataset}")
    torch.save({"arch": "cnn", "dataset": args.dataset, "classes": classes,
                "state_dict": best_state, "val_acc": best_acc, "test_acc": test_acc}, base + ".pt")
    with open(base + "_history.json", "w") as f:
        json.dump({**hist, "test_acc": test_acc}, f, indent=2)
    plot_history(hist, base + "_curves.png")
    cm = confusion_matrix(preds, targets, len(classes))
    plot_confusion(cm, classes, base + "_confusion.png")
    print("hardest classes:", [(c, f"{a * 100:.1f}%") for c, a in per_class_report(cm, classes)])
    print(f"saved model + plots to {args.out_dir}/ (prefix cnn_{args.dataset})")
    return test_acc


if __name__ == "__main__":
    main()
