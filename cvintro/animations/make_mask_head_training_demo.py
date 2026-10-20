"""Generate mask_head_training_demo.gif: an animated companion to the lesson
43 notebook's "A mask head" section. Watches the mask head learn, on the same
four held-out test crops as the book's static figure, to separate this
instance's pixels from the visibly-overlapping neighboring circle in the
same crop.

Reuses the notebook's own make_scene/make_crops/MaskHead setup verbatim
(data rng seed=17, model seed=0, 300 epochs, lr=0.01), so the final
predicted masks match the book exactly.

Run from anywhere:
    python make_mask_head_training_demo.py
Output:
    mask_head_training_demo.gif  (written next to this script)
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
OUT_GIF = os.path.join(OUT_DIR, "mask_head_training_demo.gif")

TEXT_COLOR = "#222222"
SIZE = 32


def make_scene(rng, size=SIZE, n_circles=2, r=5):
    scene = np.zeros((size, size), dtype=np.float32)
    sem_mask = np.zeros((size, size), dtype=np.int64)
    inst_mask = np.zeros((size, size), dtype=np.int64)
    intensity = {0: 0.85, 1: 1.0}
    centers = []
    for k in range(n_circles):
        if k == 0:
            cx, cy = rng.integers(r + 8, size - r - 8), rng.integers(r + 8, size - r - 8)
        else:
            px, py = centers[-1]
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(6, 9)
            cx = int(np.clip(px + dist * np.cos(angle), r, size - r - 1))
            cy = int(np.clip(py + dist * np.sin(angle), r, size - r - 1))
        centers.append((cx, cy))
        yy, xx = np.mgrid[0:size, 0:size]
        m = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2
        scene[m] = intensity[k]
        sem_mask[m] = 1
        inst_mask[m] = k + 1
    scene = np.clip(scene + rng.normal(0, 0.05, scene.shape), 0, 1).astype(np.float32)
    return scene, sem_mask, inst_mask, centers


rng = np.random.default_rng(17)
N = 300
scenes, sem_masks, inst_masks, all_centers = [], [], [], []
for _ in range(N):
    s, sm, im, c = make_scene(rng)
    scenes.append(s); sem_masks.append(sm); inst_masks.append(im); all_centers.append(c)
scenes = np.array(scenes, dtype=np.float32)

split = int(0.85 * N)

CROP = 14
HALF = CROP // 2


def make_crops(indices):
    crops, masks = [], []
    for i in indices:
        scene, inst = scenes[i], inst_masks[i]
        for k, (cx, cy) in enumerate(all_centers[i]):
            x0 = int(np.clip(cx - HALF, 0, SIZE - CROP))
            y0 = int(np.clip(cy - HALF, 0, SIZE - CROP))
            crop = scene[y0:y0 + CROP, x0:x0 + CROP]
            inst_crop = inst[y0:y0 + CROP, x0:x0 + CROP]
            mask = (inst_crop == (k + 1)).astype(np.float32)
            crops.append(crop); masks.append(mask)
    return np.array(crops, dtype=np.float32), np.array(masks, dtype=np.float32)


X_crop_train, M_crop_train = make_crops(range(split))
X_crop_test, M_crop_test = make_crops(range(split, N))
print(f"{len(X_crop_train)} training crops, {len(X_crop_test)} test crops, each {CROP}x{CROP}")


class MaskHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


EPOCHS = 300
CHECKPOINTS = sorted(set([0, 1, 2, 3, 5, 8, 12, 18, 26, 36, 50, 70, 95, 130, 170, 220, 270, EPOCHS - 1]))

print("training the mask head...")
torch.manual_seed(0)
mask_head = MaskHead()
opt = torch.optim.Adam(mask_head.parameters(), lr=0.01)
Xc = torch.tensor(X_crop_train).unsqueeze(1)
Mc = torch.tensor(M_crop_train).unsqueeze(1)
Xc_test4 = torch.tensor(X_crop_test[:4]).unsqueeze(1)

snapshots = []
for epoch in range(EPOCHS):
    if epoch in CHECKPOINTS:
        with torch.no_grad():
            pred4 = (mask_head(Xc_test4) > 0).float().squeeze(1).numpy()
        snapshots.append((epoch, pred4))
    opt.zero_grad()
    loss = F.binary_cross_entropy_with_logits(mask_head(Xc), Mc)
    loss.backward()
    opt.step()

with torch.no_grad():
    pred4 = (mask_head(Xc_test4) > 0).float().squeeze(1).numpy()
snapshots.append((EPOCHS, pred4))


def mean_iou4(pred, true):
    inter = (pred * true).sum(axis=(1, 2))
    union = ((pred + true) > 0).sum(axis=(1, 2))
    return (inter / np.clip(union, 1, None))


final_iou = mean_iou4(snapshots[-1][1], M_crop_test[:4])
print(f"final per-crop IoU on these 4 crops: {final_iou.round(3).tolist()}")


def render(epoch, pred4):
    fig, axes = plt.subplots(3, 4, figsize=(9, 5.6), dpi=120)
    fig.subplots_adjust(top=0.80, bottom=0.03, left=0.03, right=0.97, hspace=0.35, wspace=0.08)

    ious = mean_iou4(pred4, M_crop_test[:4])
    for i in range(4):
        axes[0, i].imshow(X_crop_test[i], cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(M_crop_test[i], cmap="viridis", vmin=0, vmax=1); axes[1, i].axis("off")
        axes[2, i].imshow(pred4[i], cmap="viridis", vmin=0, vmax=1); axes[2, i].axis("off")
        axes[2, i].text(0.5, -0.1, f"IoU={ious[i]:.2f}", transform=axes[2, i].transAxes, ha="center",
                         fontsize=8.5, color=TEXT_COLOR)
    for r, name in enumerate(["crop (neighbor visible)", "ground truth (this instance)", "predicted mask"]):
        axes[r, 0].set_title(name, fontsize=9, loc="left", color=TEXT_COLOR)

    fig.suptitle(f"training the mask head — epoch {epoch:3d} / {EPOCHS}   mean IoU = {ious.mean():.2f}",
                 fontsize=12.5, fontweight="bold", color=TEXT_COLOR, y=0.98)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
for i, (epoch, pred4) in enumerate(snapshots):
    is_last = (i == len(snapshots) - 1)
    hold = 90 if is_last else 10
    frames += [render(epoch, pred4)] * hold

frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=55, loop=0)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
