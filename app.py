"""Draw-and-recognise desktop demo (Tkinter - ships with Python).

    python app.py --cnn models/cnn_emnist-balanced.pt --crnn models/crnn_emnist-letters.pt

Modes
    Character      : draw ONE character, CNN shows its top-3 guesses
    Word (CNN)     : write a line with gaps between letters, it is segmented + classified
    Word (CRNN)    : write a whole word, CRNN reads it without segmentation
"""
import argparse
import os
import tkinter as tk
from tkinter import ttk

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.predict import load_model, predict_char, recognise_crnn, recognise_segmented

W, H, BRUSH = 640, 240, 12


class App:
    def __init__(self, root, cnn_path, crnn_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn = self.crnn = None
        if cnn_path and os.path.exists(cnn_path):
            self.cnn, self.cnn_ck = load_model(cnn_path, self.device)
        if crnn_path and os.path.exists(crnn_path):
            self.crnn, self.crnn_ck = load_model(crnn_path, self.device)

        root.title("Handwritten Character Recognition")
        self.canvas = tk.Canvas(root, width=W, height=H, bg="white", cursor="pencil", relief="sunken", bd=2)
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "last", None))
        self.last = None
        self.img = Image.new("L", (W, H), 255)
        self.pen = ImageDraw.Draw(self.img)

        bar = ttk.Frame(root)
        bar.pack(pady=(0, 6))
        self.mode = tk.StringVar(value="char")
        for text, val in (("Character (CNN)", "char"), ("Word (CNN + segmentation)", "segment"),
                          ("Word (CRNN)", "crnn")):
            ttk.Radiobutton(bar, text=text, value=val, variable=self.mode).pack(side="left", padx=6)
        ttk.Button(bar, text="Recognise", command=self.recognise).pack(side="left", padx=6)
        ttk.Button(bar, text="Clear", command=self.clear).pack(side="left")
        self.result = tk.StringVar(value="Draw something, then press Recognise")
        tk.Label(root, textvariable=self.result, font=("Helvetica", 18), justify="left").pack(pady=(0, 10))

    def draw(self, e):
        if self.last:
            x0, y0 = self.last
            self.canvas.create_line(x0, y0, e.x, e.y, width=BRUSH, capstyle="round", fill="black")
            self.pen.line([x0, y0, e.x, e.y], fill=0, width=BRUSH)
        r = BRUSH // 2
        self.canvas.create_oval(e.x - r, e.y - r, e.x + r, e.y + r, fill="black", outline="black")
        self.pen.ellipse([e.x - r, e.y - r, e.x + r, e.y + r], fill=0)
        self.last = (e.x, e.y)

    def clear(self):
        self.canvas.delete("all")
        self.pen.rectangle([0, 0, W, H], fill=255)
        self.result.set("Draw something, then press Recognise")

    def recognise(self):
        gray = np.asarray(self.img).copy()
        if gray.min() == 255:
            self.result.set("Nothing drawn yet")
            return
        mode = self.mode.get()
        if mode == "crnn":
            if self.crnn is None:
                self.result.set("No CRNN model loaded (train one with src.train_crnn)")
                return
            self.result.set(f"Word: {recognise_crnn(self.crnn, self.crnn_ck['charset'], gray, self.device)}")
            return
        if self.cnn is None:
            self.result.set("No CNN model loaded (train one with src.train)")
            return
        classes = self.cnn_ck["classes"]
        if mode == "char":
            top = predict_char(self.cnn, classes, gray, self.device, 3)
            self.result.set("\n".join(f"{c}   {p * 100:.1f}%" for c, p in top))
        else:
            self.result.set(f"Text: {recognise_segmented(self.cnn, classes, gray, self.device)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", default="models/cnn_emnist-balanced.pt")
    ap.add_argument("--crnn", default="models/crnn_emnist-letters.pt")
    a = ap.parse_args()
    root = tk.Tk()
    App(root, a.cnn, a.crnn)
    root.mainloop()
