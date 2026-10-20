"""Generate data_parallel_demo.gif: an animated schematic companion to the
lesson 44 notebook's "Data parallelism: correctness proven in miniature"
section. Shows the 64-sample batch splitting into 4 shards, each shard's
gradient computed independently on its own "device," the per-shard
gradients converging into an averaged gradient, and that average compared
against the true single-device (whole-batch) gradient.

Reuses the notebook's own data-parallel demo verbatim (model seed=1, same
X/y, 4 shards, deepcopy per shard), so the final reported max-abs-difference
matches the notebook's floating-point-roundoff result exactly.

Run from anywhere:
    python make_data_parallel_demo.py
Output:
    data_parallel_demo.gif  (written next to this script)
"""
import copy
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
OUT_GIF = os.path.join(OUT_DIR, "data_parallel_demo.gif")

TEXT_COLOR = "#222222"
SAMPLE_COLOR = "#9fb8d9"
SHARD_COLORS = ["#1f6fb4", "#2e9e44", "#f2994a", "#9b59b6"]
AVG_COLOR = "#555555"
MATCH_COLOR = "#2e9e44"

torch.manual_seed(1)
model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
X = torch.randn(64, 8)
y = torch.randn(64, 1)

model.zero_grad()
F.mse_loss(model(X), y).backward()
grad_single_device = [p.grad.clone() for p in model.parameters()]

n_shards = 4
shard_size = len(X) // n_shards
per_shard_grads = []
for s in range(n_shards):
    shard_model = copy.deepcopy(model)
    shard_model.zero_grad()
    xs, ys = X[s * shard_size:(s + 1) * shard_size], y[s * shard_size:(s + 1) * shard_size]
    F.mse_loss(shard_model(xs), ys).backward()
    per_shard_grads.append([p.grad.clone() for p in shard_model.parameters()])

averaged_grad = [sum(shard[i] for shard in per_shard_grads) / n_shards for i in range(len(grad_single_device))]

max_diff = max((g_single - g_avg).abs().max().item() for g_single, g_avg in zip(grad_single_device, averaged_grad))
print(f"max abs diff across all parameters: {max_diff:.2e}")

# A simple scalar "gradient signature" per device for the diagram: mean abs
# gradient of layer 0's weight. The real correctness check above uses every
# parameter exactly; this is only a visual summary.
shard_signatures = [shard[0].abs().mean().item() for shard in per_shard_grads]
single_signature = grad_single_device[0].abs().mean().item()
avg_signature = averaged_grad[0].abs().mean().item()

DEVICE_W, DEVICE_H = 1.7, 1.0
DEVICE_X = [0.5, 2.85, 5.2, 7.55]
DEVICE_Y = 5.0
BATCH_Y = 7.5
DEVICE_CENTERS = [x + DEVICE_W / 2 for x in DEVICE_X]
AVG_W, AVG_H = 1.7, 0.9
AVG_X, AVG_Y = sum(DEVICE_CENTERS) / len(DEVICE_CENTERS) - AVG_W / 2, 2.3
CMP_W, CMP_GAP = 2.0, 0.3
CMP_Y = 0.2
CMP_X0 = sum(DEVICE_CENTERS) / len(DEVICE_CENTERS) - CMP_W - CMP_GAP / 2
CMP_X1 = CMP_X0 + CMP_W + CMP_GAP

BEATS = ["intro", "split", "compute", "reduce", "compare"]
HOLD = {"intro": 35, "split": 40, "compute": 45, "reduce": 40, "compare": 90}


def draw_box(ax, x, y, w, h, color, label, lw=2.2, fill=None):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                                         edgecolor=color, facecolor=fill or "none", linewidth=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.5,
            color=TEXT_COLOR, fontweight="bold", zorder=4)


CENTER_X = sum(DEVICE_CENTERS) / len(DEVICE_CENTERS)


def render(beat):
    fig, ax = plt.subplots(figsize=(9.6, 6.8), dpi=120)
    fig.subplots_adjust(top=0.90, bottom=0.03, left=0.02, right=0.98)
    ax.set_xlim(0, 9.6)
    ax.set_ylim(-0.6, 9.3)
    ax.axis("off")

    # the 64-sample batch: 4 rows of 16 dots, one row per eventual shard
    row_ys = [BATCH_Y + s * 0.4 for s in range(4)]
    rng_x, rng_y = np.meshgrid(np.linspace(CENTER_X - 3.6, CENTER_X + 3.6, 16), row_ys)
    xs, ys = rng_x.flatten(), rng_y.flatten()
    if beat == "intro":
        colors = [SAMPLE_COLOR] * 64
    else:
        colors = []
        for s in range(n_shards):
            colors += [SHARD_COLORS[s]] * shard_size
    ax.scatter(xs, ys, s=22, color=colors, zorder=5)
    ax.text(CENTER_X, row_ys[-1] + 0.45, "batch of 64 samples", ha="center", fontsize=10,
            color=TEXT_COLOR, fontweight="bold")

    if beat == "intro":
        ax.text(CENTER_X, 4.0, "one device would compute the gradient over all 64 samples at once",
                fontsize=9.5, color=TEXT_COLOR, ha="center", style="italic")

    if beat in ("split", "compute", "reduce", "compare"):
        for s in range(n_shards):
            dx, dy = DEVICE_X[s], DEVICE_Y
            cx = dx + DEVICE_W / 2
            ax.annotate("", xy=(cx, dy + DEVICE_H), xytext=(cx, row_ys[s] - 0.2),
                        arrowprops=dict(arrowstyle="-|>", color=SHARD_COLORS[s], lw=1.6), zorder=2)
            draw_box(ax, dx, dy, DEVICE_W, DEVICE_H, SHARD_COLORS[s], f"device {s}\n(16 samples)")

    if beat in ("compute", "reduce", "compare"):
        for s in range(n_shards):
            cx = DEVICE_X[s] + DEVICE_W / 2
            bar_h = min(shard_signatures[s] * 6, 1.6)
            ax.add_patch(patches.Rectangle((cx - 0.12, DEVICE_Y - 1.3), 0.24, bar_h,
                                            color=SHARD_COLORS[s], zorder=3))
            ax.text(cx, DEVICE_Y - 1.5, f"grad\n{shard_signatures[s]:.3f}", ha="center",
                    va="top", fontsize=7.6, color=SHARD_COLORS[s], fontweight="bold")

    if beat in ("reduce", "compare"):
        for s in range(n_shards):
            cx = DEVICE_X[s] + DEVICE_W / 2
            ax.annotate("", xy=(AVG_X + AVG_W / 2, AVG_Y + AVG_H / 2), xytext=(cx, DEVICE_Y - 1.7),
                        arrowprops=dict(arrowstyle="-|>", color=SHARD_COLORS[s], lw=1.4, alpha=0.8), zorder=2)
        draw_box(ax, AVG_X, AVG_Y, AVG_W, AVG_H, AVG_COLOR, f"average\n{avg_signature:.3f}")

    if beat == "compare":
        match = abs(avg_signature - single_signature) < 1e-4
        cmp_color = MATCH_COLOR if match else "#d62728"
        fill = "#eafaf0" if match else None
        draw_box(ax, CMP_X0, CMP_Y, CMP_W, 0.9, cmp_color, f"averaged-shard\n{avg_signature:.3f}", fill=fill)
        draw_box(ax, CMP_X1, CMP_Y, CMP_W, 0.9, cmp_color, f"single-device\n{single_signature:.3f}", fill=fill)
        ax.annotate("", xy=(CENTER_X, CMP_Y + 0.9), xytext=(AVG_X + AVG_W / 2, AVG_Y - 0.1),
                    arrowprops=dict(arrowstyle="-|>", color=AVG_COLOR, lw=1.6), zorder=2)
        ax.text(CENTER_X, -0.5, f"max abs diff across every parameter: {max_diff:.1e}  (floating-point roundoff, not a real discrepancy)",
                ha="center", fontsize=8.8, color=cmp_color, fontweight="bold")

    TITLES = {
        "intro": "data parallelism: splitting a batch across devices",
        "split": "the batch splits into 4 shards, one per device",
        "compute": "each device computes its own gradient, independently",
        "reduce": "the per-device gradients are averaged (the all-reduce)",
        "compare": "averaging per-shard gradients matches the whole-batch gradient",
    }
    ax.set_title(TITLES[beat], fontsize=12.5, fontweight="bold", color=TEXT_COLOR, pad=10)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
for beat in BEATS:
    frames += [render(beat)] * HOLD[beat]

frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=55, loop=0)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
