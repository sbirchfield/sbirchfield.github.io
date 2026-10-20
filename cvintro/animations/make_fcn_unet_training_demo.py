"""Generate fcn_unet_training_demo.gif: an animated companion to the lesson 42
notebook's "Adding skip connections: U-Net" section. Trains FCNTiny and
UNetTiny on the same 200 epochs the notebook uses, capturing checkpoint
predictions on four held-out test scenes, and watches both decoders refine
their masks side by side as training progresses -- ending on the same two
rows of predictions as the notebook's static 4x4 comparison figure.

Reuses the notebook's own make_scene/FCNTiny/UNetTiny/mean_iou setup
verbatim (data rng seed=13, model seed=0 for both models, Adam lr=0.01,
200 epochs, cross-entropy loss), so the final predictions match the book
exactly.

Run from anywhere:
    python make_fcn_unet_training_demo.py
Output:
    fcn_unet_training_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "fcn_unet_training_demo.gif")

TEXT_COLOR = "#222222"


def make_scene(rng, size=32, n_shapes=5):
    scene = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.int64)  # 0=background, 1=circle, 2=square
    intensity = {1: 0.85, 2: 1.0}
    for _ in range(n_shapes):
        shape_type = rng.choice([1, 2])
        r = rng.integers(2, 4)
        cx, cy = rng.integers(r, size - r), rng.integers(r, size - r)
        yy, xx = np.mgrid[0:size, 0:size]
        if shape_type == 1:
            m = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2
        else:
            m = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
        scene[m] = intensity[shape_type]
        mask[m] = shape_type
    scene = np.clip(scene + rng.normal(0, 0.05, scene.shape), 0, 1).astype(np.float32)
    return scene, mask


rng = np.random.default_rng(13)
N = 300
scenes, masks = [], []
for _ in range(N):
    s, m = make_scene(rng)
    scenes.append(s); masks.append(m)
scenes = np.array(scenes, dtype=np.float32)
masks = np.array(masks, dtype=np.int64)

split = int(0.85 * N)
Xtr, Mtr = scenes[:split], masks[:split]
Xte, Mte = scenes[split:], masks[split:]


class FCNTiny(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(16, n_classes, 1)

    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(self.pool(f1))
        f3 = self.enc3(self.pool(f2))
        d2 = self.dec2(self.up(f3))
        d1 = self.dec1(self.up(d2))
        return self.out(d1)


class UNetTiny(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = nn.Sequential(nn.Conv2d(64 + 32, 32, 3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(32 + 16, 16, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(16, n_classes, 1)

    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(self.pool(f1))
        f3 = self.enc3(self.pool(f2))
        d2 = self.dec2(torch.cat([self.up(f3), f2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), f1], dim=1))
        return self.out(d1)


def mean_iou(preds, targets, n_classes=3):
    ious = []
    for c in range(n_classes):
        p, t = preds == c, targets == c
        inter, union = (p & t).sum(), (p | t).sum()
        ious.append(inter / union if union > 0 else float("nan"))
    return ious


EPOCHS = 200
CHECKPOINTS = sorted(set([0, 1, 2, 4, 8, 16, 32, 64, 100, 150, EPOCHS - 1]))

Xt = torch.tensor(Xtr).unsqueeze(1); Mt = torch.tensor(Mtr)
Xte4 = torch.tensor(Xte[:4]).unsqueeze(1)
Mte4 = Mte[:4]


def train_with_snapshots(model_cls):
    torch.manual_seed(0)
    model = model_cls()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    snapshots = []
    for epoch in range(EPOCHS):
        if epoch in CHECKPOINTS:
            with torch.no_grad():
                pred4 = model(Xte4).argmax(1).numpy()
            snapshots.append((epoch, pred4))
        opt.zero_grad()
        loss = F.cross_entropy(model(Xt), Mt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred4 = model(Xte4).argmax(1).numpy()
    snapshots.append((EPOCHS, pred4))

    with torch.no_grad():
        preds_full = model(torch.tensor(Xte).unsqueeze(1)).argmax(1).numpy()
    ious = mean_iou(preds_full, Mte)
    print(f"{model_cls.__name__}: pixel acc {np.mean(preds_full == Mte):.1%}, "
          f"mean IoU {np.mean(ious):.3f}")
    return snapshots


print("training FCNTiny (no skip connections)...")
fcn_snapshots = train_with_snapshots(FCNTiny)
print("training UNetTiny (with skip connections)...")
unet_snapshots = train_with_snapshots(UNetTiny)

assert [e for e, _ in fcn_snapshots] == [e for e, _ in unet_snapshots]
CHECKPOINT_EPOCHS = [e for e, _ in fcn_snapshots]


def render(idx):
    epoch = CHECKPOINT_EPOCHS[idx]
    fcn_pred4 = fcn_snapshots[idx][1]
    unet_pred4 = unet_snapshots[idx][1]

    fig, axes = plt.subplots(4, 4, figsize=(9, 9), dpi=110)
    fig.subplots_adjust(top=0.90, bottom=0.02, left=0.02, right=0.98, hspace=0.12, wspace=0.08)

    for i in range(4):
        axes[0, i].imshow(Xte[i], cmap="gray")
        axes[1, i].imshow(Mte[i], cmap="viridis", vmin=0, vmax=2)
        axes[2, i].imshow(fcn_pred4[i], cmap="viridis", vmin=0, vmax=2)
        axes[3, i].imshow(unet_pred4[i], cmap="viridis", vmin=0, vmax=2)
        for r in range(4):
            axes[r, i].axis("off")

    row_names = ["input", "ground truth mask", "pred (FCN, no skip)", "pred (U-Net, skip)"]
    for r, name in enumerate(row_names):
        axes[r, 0].set_title(name, fontsize=9, loc="left", color=TEXT_COLOR)

    fig.suptitle(f"training FCN vs. U-Net  —  epoch {epoch:3d} / {EPOCHS}", fontsize=13,
                 fontweight="bold", color=TEXT_COLOR, y=0.975)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
for i in range(len(CHECKPOINT_EPOCHS)):
    is_last = (i == len(CHECKPOINT_EPOCHS) - 1)
    hold = 90 if is_last else 10
    frames += [render(i)] * hold

frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=55, loop=0)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
