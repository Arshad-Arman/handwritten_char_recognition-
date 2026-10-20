#!/usr/bin/env python3
"""One-command launcher for the whole project.

    python run.py                 # standard pipeline (downloads EMNIST the first time)
    python run.py --mode quick    # ~5 min, no download (font-rendered data) - good for a first check
    python run.py --mode fast     # real EMNIST, only 3 epochs - good for a slow CPU
    python run.py --mode full     # longer training, best accuracy
    python run.py --skip-train    # reuse models already in ./models
    python run.py --dry-run       # only print what would be executed

Pipeline: 1 dependencies -> 2 tests -> 3 train CNN -> 4 evaluate -> 5 train CRNN
          -> 6 sample predictions -> 7 open the drawing app
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

# mode -> (CNN dataset, CNN epochs, CRNN source, CRNN epochs, CRNN words/epoch)
MODES = {
    "quick":    ("synthetic",       4, "synthetic",      6, 2048),
    "fast":     ("emnist-balanced", 3, "emnist-letters", 3, 4000),   # real EMNIST, short training
    "standard": ("emnist-balanced", 10, "emnist-letters", 10, 10000),
    "full":     ("emnist-balanced", 20, "emnist-letters", 25, 20000),
}
REQUIRED = {"torch": "torch", "torchvision": "torchvision", "numpy": "numpy",
            "cv2": "opencv-python", "matplotlib": "matplotlib", "PIL": "Pillow"}


def banner(n: int, total: int, text: str) -> None:
    print(f"\n{'=' * 70}\n[{n}/{total}] {text}\n{'=' * 70}", flush=True)


def run(cmd: list[str], dry: bool) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if dry:
        return
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # show progress lines immediately
    if subprocess.call(cmd, cwd=ROOT, env=env) != 0:
        sys.exit(f"\nStep failed: {' '.join(cmd)}\nFix the error above and run `python run.py` again.")


def missing_packages() -> list[str]:
    return [pip for mod, pip in REQUIRED.items() if importlib.util.find_spec(mod) is None]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the complete Handwritten Character Recognition project")
    ap.add_argument("--mode", choices=list(MODES), default="standard")
    ap.add_argument("--skip-train", action="store_true", help="reuse existing models in ./models")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--no-app", action="store_true", help="do not open the drawing window at the end")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args()

    cnn_ds, cnn_ep, crnn_src, crnn_ep, crnn_words = MODES[args.mode]
    cnn_model = f"models/cnn_{cnn_ds}.pt"
    crnn_model = f"models/crnn_{crnn_src}.pt"
    total, t0 = 7, time.time()
    print(f"Handwritten Character Recognition | mode: {args.mode} | python {sys.version.split()[0]}")
    if args.mode != "quick" and not args.skip_train:
        print("Note: the first run downloads EMNIST (~560 MB). Use --mode quick to skip downloads.")

    banner(1, total, "Checking dependencies")
    missing = missing_packages()
    if missing:
        print("missing:", ", ".join(missing))
        run([PY, "-m", "pip", "install", "-r", "requirements.txt"], args.dry_run)
    else:
        print("all required packages are installed")

    banner(2, total, "Running tests")
    if args.skip_tests:
        print("skipped")
    else:
        run([PY, "tests/test_vision.py"], args.dry_run)

    banner(3, total, f"Training CNN on {cnn_ds} ({cnn_ep} epochs)")
    if args.skip_train:
        print("skipped (--skip-train)")
    else:
        run([PY, "-m", "src.train", "--dataset", cnn_ds, "--epochs", str(cnn_ep)], args.dry_run)

    banner(4, total, "Evaluating CNN on the test set (confusion matrix saved to ./models)")
    run([PY, "-m", "src.evaluate", "--model", cnn_model], args.dry_run)

    banner(5, total, f"Training CRNN for word recognition ({crnn_src}, {crnn_ep} epochs)")
    if args.skip_train:
        print("skipped (--skip-train)")
    else:
        run([PY, "-m", "src.train_crnn", "--source", crnn_src, "--epochs", str(crnn_ep),
             "--train-words", str(crnn_words)], args.dry_run)

    banner(6, total, "Sample predictions")
    run([PY, "-m", "src.predict", "--image", "samples/char_A.png", "--model", cnn_model], args.dry_run)
    run([PY, "-m", "src.predict", "--image", "samples/sentence_HI_THERE.png", "--model", cnn_model,
         "--mode", "segment"], args.dry_run)
    run([PY, "-m", "src.predict", "--image", "samples/word_HELLO.png", "--model", crnn_model], args.dry_run)

    banner(7, total, "Drawing app")
    if args.no_app:
        print("skipped (--no-app)")
    else:
        print("Opening the window - draw a character, choose a mode, press Recognise.")
        run([PY, "app.py", "--cnn", cnn_model, "--crnn", crnn_model], args.dry_run)

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Models and plots are in the 'models' folder.")


if __name__ == "__main__":
    main()
