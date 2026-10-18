"""Generate overfitting_demo.gif: an animated companion to the lesson 35
notebook's "Watching it overfit" section. The static notebook cell already
plots the final train/val loss curves side by side; this animation draws
them growing epoch by epoch instead, with a live marker on each curve and
the "best val loss" vertical line only appearing once training has actually
passed that epoch -- so the moment validation loss bottoms out and starts
climbing again happens live, rather than being visible from the first
frame.

Reuses the notebook's own dataset generation (data_rng seed=2), CNN
architecture, and train_tracked() training loop (torch.manual_seed(0),
Adam, lr=0.01, 400 epochs, no weight decay) verbatim, so the curve matches
the static cell's numbers exactly.

Run from anywhere:
    python make_overfitting_demo.py
Output:
    overfitting_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "overfitting_demo.gif")

TRAIN_COLOR = "#1f6fb4"
VAL_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
OVERFIT_COLOR = "#d62728"
NEUTRAL_COLOR = "#222222"


def make_image(shape_type, cx, cy, size=16, rng=None):
    img = np.zeros((size, size), dtype=np.float32)
    if shape_type == "plus":
        img[cy - 1:cy + 2, cx - 3:cx + 4] = 1.0
        img[cy - 3:cy + 4, cx - 1:cx + 2] = 1.0
    else:
        yy, xx = np.mgrid[0:size, 0:size]
        img[((xx - cx) ** 2 + (yy - cy) ** 2) <= 9] = 1.0
    if rng is not None:
        img = np.clip(img + rng.normal(0, 0.4, img.shape), 0, 1).astype(np.float32)
    return img


def make_dataset(rng_local, n, position_range):
    imgs, labels = [], []
    for _ in range(n):
        shape_type = rng_local.choice(["plus", "circle"])
        cx, cy = rng_local.integers(*position_range), rng_local.integers(*position_range)
        imgs.append(make_image(shape_type, cx, cy, rng=rng_local))
        labels.append(0.0 if shape_type == "plus" else 1.0)
    return np.array(imgs, dtype=np.float32), np.array(labels, dtype=np.float32)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1)).squeeze(-1)


def train_tracked(model_cls, Xtr, ytr, Xval, yval, epochs=400, lr=0.01, weight_decay=0.0, seed=0):
    torch.manual_seed(seed)
    model = model_cls()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses, val_losses = [], []
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(Xtr), ytr)
        loss.backward()
        opt.step()
        with torch.no_grad():
            train_losses.append(loss.item())
            val_losses.append(F.binary_cross_entropy_with_logits(model(Xval), yval).item())
    return train_losses, val_losses


data_rng = np.random.default_rng(2)
X_train, y_train = make_dataset(data_rng, 12, (3, 13))
X_val, y_val = make_dataset(data_rng, 150, (3, 13))

Xtr_t = torch.tensor(X_train).unsqueeze(1)
ytr_t = torch.tensor(y_train)
Xval_t = torch.tensor(X_val).unsqueeze(1)
yval_t = torch.tensor(y_val)

train_losses, val_losses = train_tracked(CNN, Xtr_t, ytr_t, Xval_t, yval_t)
EPOCHS = len(train_losses)
BEST_EPOCH = int(np.argmin(val_losses))
BEST_VAL = val_losses[BEST_EPOCH]
Y_MAX = max(max(train_losses), max(val_losses)) * 1.08

print(f"best val loss {BEST_VAL:.3f} at epoch {BEST_EPOCH} (of {EPOCHS}); "
      f"final train={train_losses[-1]:.3f} val={val_losses[-1]:.3f}")


def render(epoch):
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=120)

    xs = list(range(epoch + 1))
    ax.plot(xs, train_losses[:epoch + 1], color=TRAIN_COLOR, linewidth=1.6, label="train loss")
    ax.plot(xs, val_losses[:epoch + 1], color=VAL_COLOR, linewidth=1.6, label="val loss")
    ax.scatter([epoch], [train_losses[epoch]], color=TRAIN_COLOR, s=36, zorder=5)
    ax.scatter([epoch], [val_losses[epoch]], color=VAL_COLOR, s=36, zorder=5)

    reached_best = epoch >= BEST_EPOCH
    if reached_best:
        ax.axvline(BEST_EPOCH, color=NEUTRAL_COLOR, linestyle="--", linewidth=1,
                    label="best val loss")

    past_best_margin = epoch > BEST_EPOCH and val_losses[epoch] > BEST_VAL * 1.15
    if past_best_margin:
        status, color = "overfitting: val loss climbing past its best", OVERFIT_COLOR
    elif reached_best:
        status, color = "val loss just hit its minimum here", MATCH_COLOR
    else:
        status, color = "training loss falling steadily; val loss still noisy", NEUTRAL_COLOR

    ax.set_xlim(0, EPOCHS - 1)
    ax.set_ylim(0, Y_MAX)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"epoch {epoch:3d}   train={train_losses[epoch]:.3f}   "
                 f"val={val_losses[epoch]:.3f}", fontsize=10)
    fig.suptitle(status, fontsize=11.5, color=color, fontweight="bold")
    fig.subplots_adjust(top=0.86, bottom=0.12)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
CHECKPOINT_EVERY = 4
HOLD_STEP = 2
HOLD_END = 70

checkpoints = list(range(0, EPOCHS, CHECKPOINT_EVERY))
if checkpoints[-1] != EPOCHS - 1:
    checkpoints.append(EPOCHS - 1)

for epoch in checkpoints:
    frames += [render(epoch)] * HOLD_STEP
frames += [frames[-1]] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=55,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
