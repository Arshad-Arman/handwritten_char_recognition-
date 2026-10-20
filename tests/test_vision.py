"""Tests for the image-processing half (no PyTorch needed).

    python tests/test_vision.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import binarize, center_char, prepare_word, preprocess_char  # noqa: E402
from src.segment import segment_line  # noqa: E402
from src.synthetic import UPPER, generate, render_text_image  # noqa: E402


def test_synthetic():
    imgs, labels = generate(200, UPPER, seed=0)
    assert imgs.shape == (200, 28, 28) and imgs.dtype == np.uint8
    assert labels.min() == 0 and labels.max() == len(UPPER) - 1
    assert (imgs.reshape(200, -1).max(1) > 100).all(), "every glyph must contain ink"
    print("ok  synthetic data")


def test_preprocess_char():
    # black ink on white paper, and the inverse (white chalk on black board)
    paper = render_text_image("A", seed=1)
    board = 255 - paper
    for name, img in (("paper", paper), ("board", board)):
        arr = preprocess_char(img)
        assert arr.shape == (28, 28) and arr.dtype == np.float32
        assert 0.0 <= arr.min() and arr.max() <= 1.0 and arr.max() > 0.5
        ys, xs = np.nonzero(arr > 0.1)
        assert ys.max() - ys.min() <= 24 and xs.max() - xs.min() <= 24, "glyph must fit the box"
        cy, cx = (np.mean(np.nonzero(arr)[0]), np.mean(np.nonzero(arr)[1]))
        assert abs(cy - 14) < 3 and abs(cx - 14) < 3, "glyph must be roughly centred"
    # thin pen stroke (1 px line drawing of a big 'L') gets thickened
    thin = np.full((300, 300), 255, np.uint8)
    cv2.line(thin, (100, 50), (100, 250), 0, 2)
    cv2.line(thin, (100, 250), (200, 250), 0, 2)
    arr = preprocess_char(thin)
    assert (arr > 0.5).sum() > 40, "thin strokes should be thickened to MNIST-like width"
    # blank image does not crash
    assert preprocess_char(np.full((50, 50), 255, np.uint8)).sum() == 0
    print("ok  character preprocessing (paper/board/thin/blank)")


def test_segmentation():
    img = render_text_image("HELLO", seed=3)
    items = segment_line(binarize(img))
    assert len(items) == 5 and all(i is not None for i in items), f"expected 5 chars, got {len(items)}"
    items = segment_line(binarize(render_text_image("HI THERE", seed=4)))
    kinds = ["_" if i is None else "c" for i in items]
    assert "".join(kinds) == "cc_ccccc", "".join(kinds)
    for crop in items:
        if crop is not None:
            assert center_char(crop).shape == (28, 28)
    print("ok  segmentation: 'HELLO' -> 5 chars, 'HI THERE' -> 2 + space + 5")


def test_prepare_word():
    arr = prepare_word(render_text_image("WORLD", seed=5))
    assert arr.shape[0] == 32 and arr.shape[1] >= 32 and arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0
    rows = np.nonzero(arr.max(1) > 0.2)[0]
    assert 18 <= rows.max() - rows.min() + 1 <= 26, "ink height should be ~23px like EMNIST glyphs in a 32px strip"
    print("ok  word preparation for the CRNN")


if __name__ == "__main__":
    test_synthetic()
    test_preprocess_char()
    test_segmentation()
    test_prepare_word()
    print("\nall vision tests passed")
