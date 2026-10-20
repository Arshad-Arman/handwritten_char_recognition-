"""Split a line of handwriting into characters (vertical projection profile).

Works for characters that do not touch each other. Touching/cursive writing
needs the CRNN (sequence model) instead - that is exactly why it exists.
"""
from __future__ import annotations

import numpy as np


def _runs(mask: np.ndarray):
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def segment_lines(bw: np.ndarray, min_height: int = 8):
    """Split a page into text lines (horizontal projection). Returns list of crops."""
    runs = _runs((bw > 0).any(axis=1))
    merged = []
    for r in runs:  # merge i-dots / accents that sit just above their letter
        if merged and r[0] - merged[-1][1] < 0.35 * max(r[1] - r[0] + 1, merged[-1][1] - merged[-1][0] + 1):
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    return [bw[a: b + 1] for a, b in merged if b - a + 1 >= min_height]


def segment_line(bw: np.ndarray, merge_gap: int = 1, min_ink: int = 20, detect_spaces: bool = True):
    """Split one text line into character crops.

    Returns a list whose items are character crops (2D uint8) or ``None`` for a
    space between words.
    """
    runs = _runs((bw > 0).any(axis=0))
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] - 1 <= merge_gap:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    merged = [(a, b) for a, b in merged if (bw[:, a: b + 1] > 0).sum() >= min_ink]
    if not merged:
        return []
    gaps = [merged[i + 1][0] - merged[i][1] - 1 for i in range(len(merged) - 1)]
    widths = [b - a + 1 for a, b in merged]
    thr = max(2.5 * float(np.median(gaps)), 0.5 * float(np.median(widths))) if len(gaps) >= 2 else float("inf")
    items = []
    for i, (a, b) in enumerate(merged):
        items.append(bw[:, a: b + 1])
        if detect_spaces and i < len(gaps) and gaps[i] > thr:
            items.append(None)
    return items
