# Handwritten Character Recognition

**Objective:** identify handwritten digits, letters and (as an extension) whole words.
**Approach:** image processing (OpenCV) + deep learning (PyTorch).

| Component | What it does |
|---|---|
| **CNN** (`src/model.py`) | Classifies one 28x28 character. Trained on **MNIST** (digits) or **EMNIST** (digits / letters / balanced 47 classes / by-class 62 classes). |
| **Image processing** (`src/preprocess.py`) | Turns a photo or drawing into MNIST-style input: grayscale, blur, Otsu threshold, crop, stroke thickening, resize, centre by centre-of-mass. |
| **Segmentation** (`src/segment.py`) | Splits a line of text into characters (vertical projection), detects spaces, then the CNN reads each character. |
| **CRNN** (`src/model.py`) | **CNN + BiLSTM + CTC** reads a whole word at once, with no segmentation (sequence modelling). |
| **Draw app** (`app.py`) | Tkinter canvas: draw with the mouse and get predictions live. |

---

## 1. Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows      (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```

Python 3.9+ is recommended. A GPU is optional: the CNN trains in a few minutes on CPU for MNIST; EMNIST and the CRNN are noticeably faster on a GPU.

## 2. Run everything with one command

```bash
python run.py                  # standard: install deps, test, train CNN + CRNN, evaluate, demo, open app
python run.py --mode quick     # ~5 min, no dataset download (a good first check)
python run.py --mode fast      # real EMNIST but only 3 epochs (for a slow CPU)
python run.py --mode full      # longest training, best accuracy
python run.py --skip-train     # reuse models already saved in ./models
python run.py --dry-run        # show the commands without running them
```

On Windows you can also just double-click **`run.bat`** (Linux/macOS: `bash run.sh`). Add `--mode quick` after it for the fast version.

The individual commands below are only needed if you want to run a single step.

## 3. Step-by-step commands

```bash
# (a) Train the CNN on digits  (downloads MNIST, ~11 MB, first run only)
python -m src.train --dataset mnist --epochs 8

# (b) Train on digits + letters  (downloads EMNIST, ~560 MB, first run only)
python -m src.train --dataset emnist-balanced --epochs 15

# (c) Evaluate / confusion matrix
python -m src.evaluate --model models/cnn_emnist-balanced.pt

# (d) Predict from an image
python -m src.predict --image samples/char_A.png --model models/cnn_emnist-balanced.pt
python -m src.predict --image samples/word_HELLO.png --model models/cnn_emnist-balanced.pt --mode segment

# (e) Word recognition with the CRNN
python -m src.train_crnn --source emnist-letters --epochs 15
python -m src.predict --image samples/word_HELLO.png --model models/crnn_emnist-letters.pt

# (f) Draw-and-recognise window
python app.py --cnn models/cnn_emnist-balanced.pt --crnn models/crnn_emnist-letters.pt
```

No internet / want a 2-minute sanity check? Use the built-in font-rendered data:

```bash
python -m src.train --dataset synthetic --epochs 3
python -m src.train_crnn --source synthetic --epochs 5
```

Datasets available for `--dataset`: `mnist`, `emnist-digits`, `emnist-letters`, `emnist-balanced`, `emnist-byclass`, `synthetic`.

## 4. How it works

### 4.1 Image-processing pipeline (`preprocess.py`)

```
photo/drawing -> grayscale -> Gaussian blur -> Otsu threshold (ink = white, works on dark or light paper)
              -> tight bounding-box crop -> thicken thin pen strokes (to MNIST stroke width)
              -> resize longest side to 20 px (keep aspect ratio) -> paste in 28x28
              -> shift so the centre of mass is the image centre -> normalise (mean 0.1307, std 0.3081)
```

This mirrors how MNIST/EMNIST were made, so a hand-drawn character looks like the training data.

### 4.2 CNN classifier

```
Input 1x28x28
Block1: Conv3x3(32)-BN-ReLU x2, MaxPool  -> 32x14x14
Block2: Conv3x3(64)-BN-ReLU x2, MaxPool  -> 64x7x7
Block3: Conv3x3(128)-BN-ReLU x2, MaxPool -> 128x3x3
Flatten -> Dropout(0.4) -> FC 256 -> ReLU -> Dropout(0.4) -> FC num_classes
```

Training: AdamW, one-cycle learning-rate schedule, label smoothing 0.05, random affine augmentation (rotation, shift, scale, shear), best epoch picked on a 10 % validation split, final score on the official test set. Outputs go to `models/`: the checkpoint, `*_curves.png` (loss/accuracy) and `*_confusion.png`.

### 4.3 CRNN for words and sentences

```
word image (1x32xW) -> CNN feature extractor -> sequence of W/4 feature columns
                    -> 2-layer Bidirectional LSTM -> Linear(classes + blank)
                    -> CTC loss (training) / greedy CTC decoding (prediction)
```

**Why CTC?** A word image has no per-letter labels. CTC sums over every possible alignment between the W/4 time steps and the target text, so the network learns *where* the letters are by itself and the same code works for any word length.

**Training data:** `train_crnn.py` composes random "words" on the fly by placing real EMNIST character images side by side, so no word dataset is needed.

### 4.4 Two ways to read a line of text

| | Segmentation + CNN | CRNN |
|---|---|---|
| Idea | cut at gaps, classify each piece | read the whole word, CTC aligns |
| Handles touching letters | no | yes (to the extent it was trained on them) |
| Needs word-level training data | no | helpful |

## 5. Expected results

Typical figures reported for this kind of CNN (your numbers will vary slightly; check the `*_history.json` written after training):

| Dataset | Classes | Typical test accuracy |
|---|---|---|
| MNIST | 10 | about 99.3-99.6 % |
| EMNIST digits | 10 | about 99.5-99.7 % |
| EMNIST letters | 26 | about 93-95 % |
| EMNIST balanced | 47 | about 89-91 % |
| EMNIST by-class | 62 | about 85-88 % |

EMNIST accuracies are lower because several characters are genuinely ambiguous (O/0, l/I/1, S/s, C/c, ...), which shows up in the confusion matrix.

## 6. Project structure

```
handwritten_char_recognition/
├── run.py / run.bat / run.sh  # one-command launcher
├── app.py                  # draw-and-recognise GUI
├── requirements.txt
├── samples/                # demo images (made by python -m src.synthetic)
├── models/                 # checkpoints + plots are saved here
├── data/                   # MNIST / EMNIST download here
├── src/
│   ├── constants.py        # normalisation constants
│   ├── data.py             # MNIST / EMNIST / synthetic loading, augmentation
│   ├── synthetic.py        # font-rendered characters and text (for tests/demos)
│   ├── model.py            # CNN and CRNN
│   ├── train.py            # CNN training
│   ├── evaluate.py         # accuracy, confusion matrix, plots
│   ├── preprocess.py       # OpenCV pipeline
│   ├── segment.py          # line -> characters
│   ├── ctc.py              # CTC greedy decoding, edit distance
│   ├── train_crnn.py       # CRNN training with synthetic words
│   └── predict.py          # CLI + helper functions
└── tests/
    ├── test_vision.py      # preprocessing + segmentation (no PyTorch needed)
    └── test_torch.py       # trains tiny CNN + CRNN on synthetic data end to end
```

## 7. Tests

```bash
python tests/test_vision.py     # fast, image processing only
python tests/test_torch.py      # a few minutes on CPU, no downloads
```

## 8. Limitations and future work

- **Touching / cursive letters** break the projection-based segmentation. Use the CRNN, and for real cursive handwriting train it on a word dataset such as **IAM** (swap `WordDataset` for an IAM loader).
- The CRNN trained on composed EMNIST letters is a demonstration of the sequence-modelling idea; expect it to be much weaker on natural handwriting than on composed words.
- EMNIST *letters* merges upper and lower case into one class; use `emnist-byclass` or `emnist-balanced` if case matters.
- Ideas: beam-search CTC decoding with a language model / dictionary, attention or Transformer decoders (e.g. TrOCR), data augmentation with elastic distortion, a web demo (Streamlit/Flask), ONNX export for mobile.

## 9. References

- LeCun et al., *Gradient-based learning applied to document recognition* (MNIST).
- Cohen et al., *EMNIST: an extension of MNIST to handwritten letters*, 2017.
- Shi, Bai, Yao, *An End-to-End Trainable Neural Network for Image-based Sequence Recognition* (CRNN), 2015.
- Graves et al., *Connectionist Temporal Classification*, ICML 2006.
