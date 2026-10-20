"""Synthetic character / text generator (no download needed).

Characters are rendered from the DejaVu fonts that ship with matplotlib, then
randomly rotated, scaled and shifted. It is used for:
  * quick smoke tests of the whole pipeline without downloading EMNIST,
  * generating demo images (``python -m src.synthetic``).
It is NOT real handwriting - use MNIST / EMNIST for real results.
"""
from __future__ import annotations

import argparse
import glob
import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DIGITS = "0123456789"
UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@lru_cache(maxsize=1)
def get_fonts() -> tuple:
    try:
        import matplotlib

        folder = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        paths = glob.glob(os.path.join(folder, "DejaVu*.ttf"))
        # the "...Display" files only contain symbols (no letters) -> skip them
        return tuple(sorted(p for p in paths if "Display" not in os.path.basename(p)))
    except Exception:  # pragma: no cover
        return tuple()


@lru_cache(maxsize=256)
def _font(path: str, size: int):
    return ImageFont.truetype(path, size)


def render_char(ch: str, rng: np.random.Generator, out: int = 28, box: int = 20) -> np.ndarray:
    """Return a (out, out) uint8 image, white glyph on black, MNIST/EMNIST-like."""
    fonts = get_fonts()
    if fonts:
        font = _font(str(rng.choice(fonts)), int(rng.integers(40, 64)))
    else:  # pragma: no cover
        font = ImageFont.load_default()
    canvas = Image.new("L", (96, 96), 0)
    draw = ImageDraw.Draw(canvas)
    try:
        draw.text((48, 48), ch, fill=255, font=font, anchor="mm")
    except Exception:  # pragma: no cover
        draw.text((20, 20), ch, fill=255, font=font)
    canvas = canvas.rotate(float(rng.uniform(-12, 12)), resample=Image.BILINEAR)
    arr = np.asarray(canvas)
    ys, xs = np.nonzero(arr > 40)
    if len(ys) == 0:  # pragma: no cover
        return np.zeros((out, out), np.uint8)
    arr = arr[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = arr.shape
    scale = box / max(h, w) * float(rng.uniform(0.85, 1.0))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    glyph = Image.fromarray(arr).resize((nw, nh), Image.LANCZOS)
    img = Image.new("L", (out, out), 0)
    ox = (out - nw) // 2 + int(rng.integers(-2, 3))
    oy = (out - nh) // 2 + int(rng.integers(-2, 3))
    img.paste(glyph, (max(0, ox), max(0, oy)))
    return np.asarray(img, dtype=np.uint8).copy()


def generate(n: int, charset: str, seed: int = 0):
    """Return images (n, 28, 28) uint8 and labels (n,) int64, class-balanced."""
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % len(charset)
    rng.shuffle(labels)
    images = np.stack([render_char(charset[i], rng) for i in labels])
    return images, labels.astype(np.int64)


def render_text_image(text: str, seed: int = 0, height: int = 64, gap=(6, 14),
                      space: int = 46, margin: int = 16) -> np.ndarray:
    """Black text on white paper (uint8), e.g. for testing segmentation."""
    rng = np.random.default_rng(seed)
    parts = []
    for ch in text:
        if ch == " ":
            parts.append(np.zeros((height, space), np.uint8))
            continue
        g = render_char(ch, rng)
        cols = np.nonzero((g > 40).any(axis=0))[0]
        g = g[:, cols.min(): cols.max() + 1]
        new_w = max(6, round(g.shape[1] * height / 28))
        g = np.asarray(Image.fromarray(g).resize((new_w, height), Image.LANCZOS))
        parts.append(g)
        parts.append(np.zeros((height, int(rng.integers(gap[0], gap[1] + 1))), np.uint8))
    line = np.concatenate(parts[:-1], axis=1)
    line = np.pad(line, margin)
    return 255 - line


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Write demo images to ./samples")
    ap.add_argument("--out", default="samples")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    demos = {"char_A.png": "A", "char_7.png": "7", "word_HELLO.png": "HELLO",
             "word_WORLD.png": "WORLD", "sentence_HI_THERE.png": "HI THERE"}
    for i, (name, txt) in enumerate(demos.items()):
        img = render_text_image(txt, seed=i + 10)
        Image.fromarray(img).save(os.path.join(args.out, name))
        print("wrote", os.path.join(args.out, name))
