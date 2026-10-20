"""Generate fcn_encoder_decoder_schematic_demo.gif: an animated companion to
the lesson 42 notebook's "A minimal FCN-style network" section. Traces one
test image through FCNTiny's own forward pass, stage by stage, revealing how
spatial resolution shrinks through the encoder (two rounds of max-pooling)
and is upsampled back through the decoder -- the "just downsample then
upsample" recipe the text describes.

Reuses the notebook's own make_scene/FCNTiny setup verbatim (data rng
seed=13, model seed=0, Adam lr=0.01, 200 epochs), so the final per-pixel
prediction matches the book exactly.

Run from anywhere:
    python make_fcn_encoder_decoder_schematic_demo.py
Output:
    fcn_encoder_decoder_schematic_demo.gif  (written next to this script)
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
OUT_GIF = os.path.join(OUT_DIR, "fcn_encoder_decoder_schematic_demo.gif")

TEXT_COLOR = "#222222"


def make_scene(rng, size=32, n_shapes=5):
    scene = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.int64)
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


print("training FCNTiny...")
torch.manual_seed(0)
model_fcn = FCNTiny()
opt = torch.optim.Adam(model_fcn.parameters(), lr=0.01)
Xt = torch.tensor(Xtr).unsqueeze(1); Mt = torch.tensor(Mtr)
for _ in range(200):
    opt.zero_grad()
    loss = F.cross_entropy(model_fcn(Xt), Mt)
    loss.backward()
    opt.step()

EXAMPLE_IDX = 0
x = torch.tensor(Xte[EXAMPLE_IDX:EXAMPLE_IDX + 1]).unsqueeze(1)

with torch.no_grad():
    f1 = model_fcn.enc1(x)
    f2 = model_fcn.enc2(model_fcn.pool(f1))
    f3 = model_fcn.enc3(model_fcn.pool(f2))
    d2 = model_fcn.dec2(model_fcn.up(f3))
    d1 = model_fcn.dec1(model_fcn.up(d2))
    logits = model_fcn.out(d1)
    probs = F.softmax(logits, dim=1)[0].numpy()  # (3, H, W), continuous, sums to 1 per pixel
    pred = logits.argmax(1)[0].numpy()

print(f"final prediction matches ground truth on this example: "
      f"{(pred == Mte[EXAMPLE_IDX]).mean():.1%} of pixels")

# Blend each pixel's 3-way softmax probability into the SAME 3 colors the
# final hard-label frame uses (sampled from viridis at the 3 class values),
# so the "soft probabilities" stage and the "hard prediction" stage share a
# color identity -- the viewer watches the blend snap to pure colors, rather
# than two unrelated-looking visualizations in a row.
_cmap = plt.cm.viridis
class_colors = np.array([_cmap(0.0)[:3], _cmap(0.5)[:3], _cmap(1.0)[:3]])  # (3, 3) RGB
soft_rgb = np.einsum("chw,cd->hwd", probs, class_colors)  # (H, W, 3)

STAGES = [
    ("input", Xte[EXAMPLE_IDX], "32x32 (full res)", "gray", None),
    ("encoder stage 1", f1[0].mean(0).numpy(), "32x32 (full res)", "magma", None),
    ("encoder stage 2", f2[0].mean(0).numpy(), "16x16 (1/2 res)", "magma", None),
    ("encoder stage 3 -- bottleneck", f3[0].mean(0).numpy(), "8x8 (1/4 res)", "magma", None),
    ("decoder stage 2 (upsampled)", d2[0].mean(0).numpy(), "16x16 (1/2 res)", "magma", None),
    ("decoder stage 1 (upsampled)", d1[0].mean(0).numpy(), "32x32 (full res)", "magma", None),
    ("class probabilities (soft)", soft_rgb, "32x32 (full res)", None, None),
    ("final per-pixel prediction (hard)", pred, "32x32 (full res)", "viridis", (0, 2)),
]


def render(stage_idx):
    fig, ax = plt.subplots(figsize=(5.6, 5.9), dpi=120)
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.08, right=0.92)

    name, data, res_label, cmap, vrange = STAGES[stage_idx]
    if data.ndim == 3:
        im = ax.imshow(data)  # already an RGB array (the soft-probability blend)
    elif vrange is not None:
        im = ax.imshow(data, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
    else:
        im = ax.imshow(data, cmap=cmap)
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("tracing one image through FCNTiny", fontsize=13, fontweight="bold",
                 color=TEXT_COLOR, y=0.97)
    ax.set_title(f"{name}\n{res_label}", fontsize=11.5, color=TEXT_COLOR, fontweight="bold", pad=10)

    # A small progress strip along the bottom shows where this stage sits in
    # the encoder-downsample / decoder-upsample pipeline.
    pax = fig.add_axes([0.12, 0.045, 0.76, 0.035])
    pax.set_xlim(0, len(STAGES) - 1); pax.set_ylim(0, 1)
    pax.axis("off")
    pax.plot([0, len(STAGES) - 1], [0.5, 0.5], color="#cccccc", linewidth=4,
              solid_capstyle="round", zorder=1)
    pax.scatter([stage_idx], [0.5], s=120, color="#1f6fb4", edgecolor="white",
                linewidth=1.2, zorder=2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
for i in range(len(STAGES)):
    is_last = (i == len(STAGES) - 1)
    hold = 70 if is_last else 32
    frames += [render(i)] * hold

frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=55, loop=0)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
