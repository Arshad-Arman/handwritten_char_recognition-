"""Image-processing pipeline (OpenCV) that turns a photo / drawing into model input.

Single character  ->  28x28 image in MNIST/EMNIST style:
    gray -> blur -> Otsu threshold (ink = white) -> tight crop -> stroke
    thickening -> resize to 20x20 box -> centre by centre-of-mass in 28x28.
Word image        ->  height-32 image for the CRNN.
"""
from __future__ import annotations

import cv2
import numpy as np

from .constants import MEAN, STD


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.shape[2] == 4:  # RGBA: composite on white
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Return uint8 mask with ink = 255 on black background (handles dark or light paper)."""
    gray = to_gray(gray)
    if int(gray.max()) - int(gray.min()) < 15:  # blank page
        return np.zeros_like(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    border = np.concatenate([blur[0], blur[-1], blur[:, 0], blur[:, -1]])
    light_paper = np.median(border) > 127
    flag = cv2.THRESH_BINARY_INV if light_paper else cv2.THRESH_BINARY
    _, bw = cv2.threshold(blur, 0, 255, flag + cv2.THRESH_OTSU)
    return bw


def _bbox_crop(bw: np.ndarray):
    ys, xs = np.nonzero(bw)
    if len(ys) == 0:
        return None
    return bw[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def _thicken(bw: np.ndarray, ref: int) -> np.ndarray:
    """Make thin pen strokes as thick (~12% of ``ref``) as MNIST/EMNIST strokes."""
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    vals = dist[dist > 0]
    if vals.size == 0:
        return bw
    stroke = 2.0 * float(np.percentile(vals, 90))
    target = 0.12 * ref
    if stroke >= target:
        return bw
    k = int(round(target - stroke)) | 1
    bw = cv2.copyMakeBorder(bw, k, k, k, k, cv2.BORDER_CONSTANT, value=0)
    return cv2.dilate(bw, np.ones((k, k), np.uint8))


def center_char(bw: np.ndarray, size: int = 28, box: int = 20) -> np.ndarray:
    """Binary char mask (ink=255) -> float32 [0,1] (size x size), MNIST-style."""
    crop = _bbox_crop(bw)
    if crop is None:
        return np.zeros((size, size), np.float32)
    crop = _bbox_crop(_thicken(crop, max(crop.shape)))
    h, w = crop.shape
    scale = box / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), np.uint8)
    y0, x0 = (size - new_h) // 2, (size - new_w) // 2
    canvas[y0: y0 + new_h, x0: x0 + new_w] = small
    m = cv2.moments(canvas)
    if m["m00"] > 0:  # shift so the centre of mass is the image centre (as in MNIST)
        shift = np.float32([[1, 0, size / 2 - m["m10"] / m["m00"]], [0, 1, size / 2 - m["m01"] / m["m00"]]])
        canvas = cv2.warpAffine(canvas, shift, (size, size))
    return canvas.astype(np.float32) / 255.0


def preprocess_char(img: np.ndarray) -> np.ndarray:
    """Any photo/drawing of ONE character -> 28x28 float32 array."""
    return center_char(binarize(img))


def to_tensor(arr: np.ndarray):
    """28x28 (or HxW) float [0,1] array -> normalised tensor (1, 1, H, W)."""
    import torch  # imported lazily so the rest of this module works without PyTorch

    x = torch.from_numpy(np.ascontiguousarray(arr)).float()
    return ((x - MEAN) / STD).unsqueeze(0).unsqueeze(0)


def prepare_word(img: np.ndarray, height: int = 32, glyph_ratio: float = 20 / 28, pad: int = 6) -> np.ndarray:
    """Word image -> float32 [0,1] array of height 32 for the CRNN.

    The ink height is scaled to ``glyph_ratio * height`` (same proportion as an
    EMNIST glyph inside its 28px canvas) and centred vertically.
    """
    bw = binarize(img)
    crop = _bbox_crop(bw)
    if crop is None:
        return np.zeros((height, 4 * height), np.float32)
    crop = _bbox_crop(_thicken(crop, crop.shape[0]))
    h, w = crop.shape
    target_h = max(4, round(height * glyph_ratio))
    scale = target_h / h
    new_w = max(8, round(w * scale))
    small = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, new_w + 2 * pad), np.uint8)
    y0 = (height - target_h) // 2
    canvas[y0: y0 + target_h, pad: pad + new_w] = small
    return canvas.astype(np.float32) / 255.0
