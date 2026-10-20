"""Train a CRNN (CNN + BiLSTM + CTC) to read whole words.

Training words are *composed on the fly* from isolated character images
(EMNIST or synthetic), so no word-level dataset is required. For real cursive
handwriting, swap ``WordDataset`` for a word dataset such as IAM.

Examples
    python -m src.train_crnn --source emnist-letters --epochs 15
    python -m src.train_crnn --source synthetic --epochs 5      # no download
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .ctc import edit_distance, greedy_decode
from .data import CLASS_MAPS, MEAN, STD, load_raw
from .model import CRNN

HEIGHT = 32


class WordDataset(Dataset):
    """Random words built by concatenating character images side by side."""

    def __init__(self, images, labels, num_classes, n_samples, min_len=2, max_len=6,
                 seed=0, deterministic=False):
        self.images, self.n, self.seed, self.det = images, n_samples, seed, deterministic
        self.min_len, self.max_len, self.k = min_len, max_len, num_classes
        self.by_label = [np.nonzero(labels == c)[0] for c in range(num_classes)]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed + i) if self.det else np.random.default_rng()
        length = int(rng.integers(self.min_len, self.max_len + 1))
        chars = rng.integers(0, self.k, size=length)
        parts = []
        for c in chars:
            g = self.images[rng.choice(self.by_label[c])]
            cols = np.nonzero((g > 30).any(axis=0))[0]
            g = g[:, cols.min(): cols.max() + 1] if len(cols) else g
            w = max(8, round(g.shape[1] * HEIGHT / 28))
            parts.append(cv2.resize(g, (w, HEIGHT), interpolation=cv2.INTER_LINEAR))
            parts.append(np.zeros((HEIGHT, int(rng.integers(4, 10))), np.uint8))
        word = np.concatenate(parts[:-1], axis=1).astype(np.float32) / 255.0
        x = torch.from_numpy((word - MEAN) / STD).unsqueeze(0)
        return x, torch.as_tensor(chars + 1, dtype=torch.long), chars


def collate_words(batch):
    widths = [b[0].shape[2] for b in batch]
    pad_value = -MEAN / STD  # normalised "black background"
    x = torch.full((len(batch), 1, HEIGHT, max(widths)), pad_value)
    for i, (img, _, _) in enumerate(batch):
        x[i, :, :, : img.shape[2]] = img
    targets = torch.cat([b[1] for b in batch])
    tgt_len = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
    in_len = torch.tensor([w // 4 for w in widths], dtype=torch.long)
    return x, targets, in_len, tgt_len, [b[2] for b in batch]


@torch.no_grad()
def validate(model, loader, charset, device):
    model.eval()
    words = correct = chars = errors = 0
    for x, _, _, _, truths in loader:
        preds = greedy_decode(model(x.to(device)), charset)
        for p, t in zip(preds, truths):
            t = "".join(charset[k] for k in t)
            words += 1
            correct += p == t
            chars += len(t)
            errors += edit_distance(p, t)
    return correct / words, errors / chars


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train a CRNN for word recognition")
    ap.add_argument("--source", default="emnist-letters", choices=list(CLASS_MAPS))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--train-words", type=int, default=20000, help="words generated per epoch")
    ap.add_argument("--val-words", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--synthetic-size", type=int, default=20000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    print(f"device: {device} | character source: {args.source}")

    tr_imgs, tr_lbl, charset = load_raw(args.source, args.data_dir, True, args.synthetic_size)
    te_imgs, te_lbl, _ = load_raw(args.source, args.data_dir, False, max(args.synthetic_size // 5, 2000))
    train_ds = WordDataset(tr_imgs, tr_lbl, len(charset), args.train_words, seed=args.seed)
    val_ds = WordDataset(te_imgs, te_lbl, len(charset), args.val_words, seed=10_000, deterministic=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate_words)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_words)

    model = CRNN(len(charset)).to(device)
    print(f"charset ({len(charset)}): {charset} | parameters: {sum(p.numel() for p in model.parameters()):,}")
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = (args.train_words + args.batch_size - 1) // args.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps)

    best_acc, best_state, best_cer = -1.0, None, 1.0
    for epoch in range(1, args.epochs + 1):
        t0, total, n = time.time(), 0.0, 0
        model.train()
        for x, targets, in_len, tgt_len, _ in train_loader:
            x = x.to(device)
            log_probs = model(x).log_softmax(2)
            loss = ctc(log_probs, targets, in_len, tgt_len)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total, n = total + loss.item(), n + 1
        acc, cer = validate(model, val_loader, charset, device)
        if acc > best_acc or (acc == best_acc and cer < best_cer):
            best_acc, best_cer = acc, cer
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:02d}/{args.epochs} | ctc loss {total / n:.3f} | "
              f"val word acc {acc * 100:.1f}% | CER {cer * 100:.1f}% | {time.time() - t0:.0f}s")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"crnn_{args.source}.pt")
    torch.save({"arch": "crnn", "source": args.source, "charset": charset, "state_dict": best_state,
                "word_acc": best_acc, "cer": best_cer}, path)
    print(f"\nBEST val word accuracy {best_acc * 100:.1f}% | CER {best_cer * 100:.1f}% -> saved {path}")
    return best_acc


if __name__ == "__main__":
    main()
