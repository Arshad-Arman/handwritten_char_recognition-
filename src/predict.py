"""Run a trained model on an image.

    # one character with the CNN
    python -m src.predict --image samples/char_A.png --model models/cnn_emnist-balanced.pt
    # a line of text: segment into characters, classify each with the CNN
    python -m src.predict --image samples/word_HELLO.png --model models/cnn_emnist-balanced.pt --mode segment
    # a whole word with the CRNN (no segmentation needed)
    python -m src.predict --image samples/word_HELLO.png --model models/crnn_emnist-letters.pt
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch

from .ctc import greedy_decode
from .model import CNN, CRNN
from .preprocess import binarize, center_char, prepare_word, preprocess_char, to_tensor
from .segment import segment_line, segment_lines


def load_model(path: str, device="cpu"):
    ck = torch.load(path, map_location=device)
    if ck["arch"] == "cnn":
        model = CNN(len(ck["classes"]))
    elif ck["arch"] == "crnn":
        model = CRNN(len(ck["charset"]))
    else:
        raise ValueError(f"unknown architecture {ck['arch']}")
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


@torch.no_grad()
def classify(model, classes: str, arr28: np.ndarray, device="cpu", topk: int = 3):
    """28x28 float array -> [(char, probability), ...] best first."""
    probs = torch.softmax(model(to_tensor(arr28).to(device)), dim=1)[0]
    top = probs.topk(min(topk, len(classes)))
    return [(classes[i], float(p)) for p, i in zip(top.values.cpu(), top.indices.cpu())]


def predict_char(model, classes, img, device="cpu", topk=3):
    return classify(model, classes, preprocess_char(img), device, topk)


def recognise_segmented(model, classes, img, device="cpu", multiline=False):
    """Line(s) of text -> string, via projection-profile segmentation + CNN."""
    bw = binarize(img)
    lines = segment_lines(bw) if multiline else [bw]
    out_lines = []
    for line in lines:
        text = ""
        for item in segment_line(line):
            text += " " if item is None else classify(model, classes, center_char(item), device, 1)[0][0]
        out_lines.append(text)
    return "\n".join(out_lines)


@torch.no_grad()
def recognise_crnn(model, charset, img, device="cpu"):
    """Word image -> string with the CRNN (greedy CTC decoding)."""
    arr = prepare_word(img)
    logits = model(to_tensor(arr).to(device))
    return greedy_decode(logits, charset)[0]


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"cannot read image: {path}")
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(description="Recognise handwritten text in an image")
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["char", "segment"], default="char",
                    help="for CNN models: one character, or a line segmented into characters")
    ap.add_argument("--multiline", action="store_true", help="segment mode: also split into text lines")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(args.model, device)
    img = read_image(args.image)
    if ck["arch"] == "crnn":
        print("prediction:", recognise_crnn(model, ck["charset"], img, device))
    elif args.mode == "segment":
        print("prediction:", recognise_segmented(model, ck["classes"], img, device, args.multiline))
    else:
        for ch, p in predict_char(model, ck["classes"], img, device, args.topk):
            print(f"  {ch!r}  {p * 100:5.1f}%")


if __name__ == "__main__":
    main()
