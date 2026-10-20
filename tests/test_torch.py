"""End-to-end test of the deep-learning half. Needs PyTorch, NO dataset download.

    python tests/test_torch.py

It trains the CNN and the CRNN for a couple of minutes on synthetic font data and
checks that the saved models can read images through the real prediction code.
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import train, train_crnn  # noqa: E402
from src.model import CNN, CRNN  # noqa: E402
from src.predict import load_model, predict_char, recognise_crnn, recognise_segmented  # noqa: E402
from src.synthetic import render_text_image  # noqa: E402


def test_shapes():
    assert CNN(47)(torch.zeros(2, 1, 28, 28)).shape == (2, 47)
    out = CRNN(26)(torch.zeros(3, 1, 32, 160))
    assert out.shape == (40, 3, 27), out.shape  # T = W // 4, K = classes + blank
    print("ok  model output shapes")


def test_cnn_pipeline(tmp):
    acc = train.main(["--dataset", "synthetic", "--epochs", "4", "--synthetic-size", "8000",
                      "--out-dir", tmp, "--batch-size", "64"])
    assert acc > 0.6, f"CNN failed to learn (test acc {acc:.2f})"
    model, ck = load_model(os.path.join(tmp, "cnn_synthetic.pt"))
    hits = 0
    for i, ch in enumerate("ABCDEHKMNRSTUWXYZ2345679"):
        top = predict_char(model, ck["classes"], render_text_image(ch, seed=100 + i), topk=1)
        hits += top[0][0] == ch
    print(f"ok  CNN trained (test acc {acc * 100:.1f}%), predict_char on fresh images: {hits}/25")
    text = recognise_segmented(model, ck["classes"], render_text_image("HELLO", seed=7))
    print(f"    segmentation + CNN read 'HELLO' as {text!r}")
    assert len(text) == 5


def test_crnn_pipeline(tmp):
    acc = train_crnn.main(["--source", "synthetic", "--epochs", "6", "--train-words", "2048",
                           "--val-words", "128", "--synthetic-size", "6000", "--out-dir", tmp])
    model, ck = load_model(os.path.join(tmp, "crnn_synthetic.pt"))
    word = recognise_crnn(model, ck["charset"], render_text_image("HELLO", seed=7))
    print(f"ok  CRNN trained (val word acc {acc * 100:.1f}%), read 'HELLO' as {word!r}")


if __name__ == "__main__":
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    with tempfile.TemporaryDirectory() as tmp:
        test_shapes()
        test_cnn_pipeline(tmp)
        test_crnn_pipeline(tmp)
    print("\nall deep-learning tests passed")
