"""Generate box_regression_training_demo.gif: an animated companion to the
lesson 41 notebook's "The model and the IoU metric" section. Watches the
predicted (red, dashed) box snap onto the true (green) box over training,
on the same 4 held-out test scenes as the book's static figure, with each
panel's IoU updating live.

Reuses the notebook's own make_scene/Detector/iou_batch/training setup
verbatim (data rng seed=9, model seed=0, 400 epochs, lr=0.005, MSE loss),
so the final frame's boxes and IoUs match the book exactly.

Run from anywhere:
    python make_box_regression_training_demo.py
Output:
    box_regression_training_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "box_regression_training_demo.gif")

TEXT_COLOR = "#222222"
TRUE_COLOR = "#2e9e44"
PRED_COLOR = "#d62728"

SIZE = 32


def make_scene(rng, size=SIZE, obj_size=8):
    scene = np.zeros((size, size), dtype=np.float32)
    cx = rng.integers(obj_size, size - obj_size)
    cy = rng.integers(obj_size, size - obj_size)
    yy, xx = np.mgrid[0:size, 0:size]
    scene[((xx - cx) ** 2 + (yy - cy) ** 2) <= (obj_size * 0.5) ** 2] = 1.0
    scene = np.clip(scene + rng.normal(0, 0.05, scene.shape), 0, 1).astype(np.float32)
    box = (cx, cy, obj_size, obj_size)
    return scene, box


rng = np.random.default_rng(9)
N = 400
scenes, boxes = [], []
for _ in range(N):
    s, b = make_scene(rng)
    scenes.append(s); boxes.append(b)
scenes = np.array(scenes, dtype=np.float32)
boxes = np.array(boxes, dtype=np.float32)

split = int(0.85 * N)
Xtr, Btr = scenes[:split], boxes[:split] / SIZE
Xte, Bte = scenes[split:], boxes[split:] / SIZE


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveMaxPool2d(1),
        )
        self.fc = nn.Linear(32, 4)

    def forward(self, x):
        return torch.sigmoid(self.fc(self.conv(x).flatten(1)))


def iou_batch(pred, target):
    pcx, pcy, pw, ph = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    tcx, tcy, tw, th = target[:, 0], target[:, 1], target[:, 2], target[:, 3]
    px0, py0, px1, py1 = pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2
    tx0, ty0, tx1, ty1 = tcx - tw / 2, tcy - th / 2, tcx + tw / 2, tcy + th / 2
    ix0, iy0 = torch.maximum(px0, tx0), torch.maximum(py0, ty0)
    ix1, iy1 = torch.minimum(px1, tx1), torch.minimum(py1, ty1)
    inter = (ix1 - ix0).clamp(min=0) * (iy1 - iy0).clamp(min=0)
    union = pw * ph + tw * th - inter
    return inter / union.clamp(min=1e-8)


EPOCHS = 400
CHECKPOINTS = sorted(set([0, 1, 2, 3, 4, 6, 8, 10, 14, 18, 24, 30, 40, 55, 70, 90,
                           120, 160, 200, 250, 300, 350, EPOCHS - 1]))

torch.manual_seed(0)
model = Detector()
opt = torch.optim.Adam(model.parameters(), lr=0.005)
Xt = torch.tensor(Xtr).unsqueeze(1); Bt = torch.tensor(Btr)
Xte4 = torch.tensor(Xte[:4]).unsqueeze(1)
Bte4 = torch.tensor(Bte[:4])

print("training the box regressor...")
snapshots = []
for epoch in range(EPOCHS):
    if epoch in CHECKPOINTS:
        with torch.no_grad():
            pred4 = model(Xte4)
            iou4 = iou_batch(pred4, Bte4)
        snapshots.append((epoch, pred4.clone(), iou4.clone()))
    opt.zero_grad()
    loss = F.mse_loss(model(Xt), Bt)
    loss.backward()
    opt.step()

with torch.no_grad():
    pred4 = model(Xte4)
    iou4 = iou_batch(pred4, Bte4)
snapshots.append((EPOCHS, pred4.clone(), iou4.clone()))
print(f"final mean IoU on these 4 scenes: {iou4.mean().item():.3f}")


def render(epoch, pred4, iou4):
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 3.1), dpi=120)
    fig.subplots_adjust(top=0.70, bottom=0.04, left=0.02, right=0.98, wspace=0.08)

    for i, ax in enumerate(axes):
        ax.imshow(Xte[i], cmap="gray", vmin=0, vmax=1)
        tb = Bte[i] * SIZE
        pb = pred4[i].numpy() * SIZE
        ax.add_patch(patches.Rectangle((tb[0] - tb[2] / 2, tb[1] - tb[3] / 2), tb[2], tb[3],
                                        edgecolor=TRUE_COLOR, facecolor="none", linewidth=2))
        ax.add_patch(patches.Rectangle((pb[0] - pb[2] / 2, pb[1] - pb[3] / 2), pb[2], pb[3],
                                        edgecolor=PRED_COLOR, facecolor="none", linewidth=1.6, linestyle="--"))
        ax.set_title(f"IoU={iou4[i]:.2f}", fontsize=9.5, color=TEXT_COLOR)
        ax.axis("off")

    fig.text(0.5, 0.865, "green = true box, red dashed = predicted box", fontsize=8.5,
              color=TEXT_COLOR, ha="center")

    fig.suptitle(f"training a box regressor  —  epoch {epoch:3d} / {EPOCHS}   "
                 f"mean IoU = {iou4.mean():.2f}", fontsize=12, fontweight="bold", color=TEXT_COLOR, y=0.99)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
for i, (epoch, pred4, iou4) in enumerate(snapshots):
    is_last = (i == len(snapshots) - 1)
    hold = 90 if is_last else 10
    frames += [render(epoch, pred4, iou4)] * hold

frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=55, loop=0)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
