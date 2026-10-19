"""Generate vanishing_gradient_demo.gif: an animated companion to the
lesson 37 notebook's "The vanishing-gradient problem" section. Reveals
gradient magnitude vs. layer depth, right to left (output to input) --
matching the actual direction backprop computes in -- for a 30-layer plain
sigmoid network vs. a residual (ResNet-style) network built the same way.

Reuses the notebook's own PlainDeepNet/ResidualDeepNet/grad_norms exactly
(DEPTH=30, WIDTH=32, xavier_normal_ init with sigmoid gain, model seed=1,
input seed=0), so the final ratios match the book's numbers
(plain ~2.07e-19, residual ~1.05) exactly. No training involved -- a single
forward+backward pass -- so this script runs in well under a second.

Run from anywhere:
    python make_vanishing_gradient_demo.py
Output:
    vanishing_gradient_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "vanishing_gradient_demo.gif")

# match the static "Gradient magnitude vs. depth" figure right above this
# animation in the notebook exactly: default matplotlib color cycle
# (C0/C1), not custom colors.
PLAIN_COLOR = "C0"
RES_COLOR = "C1"
TEXT_COLOR = "#222222"
SWEEP_COLOR = "#e07b00"

DEPTH = 30
WIDTH = 32


def make_layers(seed):
    torch.manual_seed(seed)
    layers = nn.ModuleList([nn.Linear(WIDTH, WIDTH) for _ in range(DEPTH)])
    for layer in layers:
        nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain("sigmoid"))
    return layers


class PlainDeepNet(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.layers = make_layers(seed)

    def forward(self, x):
        acts = [x]
        for layer in self.layers:
            x = torch.sigmoid(layer(x))
            x.retain_grad()
            acts.append(x)
        return x, acts


class ResidualDeepNet(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.layers = make_layers(seed)

    def forward(self, x):
        acts = [x]
        for layer in self.layers:
            x = x + 0.3 * torch.sigmoid(layer(x))
            x.retain_grad()
            acts.append(x)
        return x, acts


def grad_norms(model_cls):
    torch.manual_seed(0)
    x = torch.randn(8, WIDTH, requires_grad=True)
    model = model_cls(seed=1)
    out, acts = model(x)
    out.sum().backward()
    return np.array([a.grad.norm().item() for a in acts if a.grad is not None])


plain_norms = grad_norms(PlainDeepNet)
res_norms = grad_norms(ResidualDeepNet)
depths = np.arange(DEPTH + 1)  # 0 = input, DEPTH = last layer's output

plain_ratio = plain_norms[0] / plain_norms[-1]
res_ratio = res_norms[0] / res_norms[-1]
print(f"plain ratio (input/output): {plain_ratio:.2e}")
print(f"residual ratio (input/output): {res_ratio:.2e}")

YMIN, YMAX = min(plain_norms.min(), res_norms.min()) * 0.3, max(plain_norms.max(), res_norms.max()) * 3

# reveal order: backprop starts at the output (depth=DEPTH) and works
# backward to the input (depth=0) -- so the sweep goes right to left.
SWEEP_ORDER = list(range(DEPTH, -1, -1))


def render(step):
    is_stepping = step < len(SWEEP_ORDER)
    revealed_depth = SWEEP_ORDER[min(step, len(SWEEP_ORDER) - 1)]
    # all depths >= revealed_depth have been "computed" so far
    mask = depths >= revealed_depth

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=120)
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.15, right=0.96)

    ax.semilogy(depths[mask], plain_norms[mask], marker=".", color=PLAIN_COLOR,
                 linewidth=1.8, label="plain (sigmoid) net")
    ax.semilogy(depths[mask], res_norms[mask], marker=".", color=RES_COLOR,
                 linewidth=1.8, label="residual net")

    if is_stepping:
        ax.axvline(revealed_depth, color=SWEEP_COLOR, linewidth=1.6, linestyle="--", zorder=1)
        ax.scatter([revealed_depth], [plain_norms[revealed_depth]], color=PLAIN_COLOR,
                    s=70, zorder=5, edgecolor="white", linewidth=1.0)
        ax.scatter([revealed_depth], [res_norms[revealed_depth]], color=RES_COLOR,
                    s=70, zorder=5, edgecolor="white", linewidth=1.0)

    ax.set_xlim(-1, DEPTH + 1)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel("layer depth (0 = input)")
    ax.set_ylabel("gradient norm (log scale)")
    ax.legend()

    # title matches the static figure above this animation exactly
    # ("Gradient magnitude vs. depth"); the dynamic sweep/ratio info goes
    # in a smaller subtitle instead of replacing it.
    if is_stepping:
        subtitle = f"backprop sweeping backward -- now at depth {revealed_depth}"
        color = SWEEP_COLOR
    else:
        subtitle = f"plain {plain_ratio:.1e} (≈ 0)  vs.  residual {res_ratio:.2f}"
        color = TEXT_COLOR
    ax.set_title(subtitle, fontsize=10, color=color, fontweight="bold")
    fig.suptitle("Gradient magnitude vs. depth", fontsize=12.5, color=TEXT_COLOR, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 4
HOLD_END = 100
n_steps = len(SWEEP_ORDER)
for step in range(n_steps + 1):
    frames += [render(step)] * (HOLD_END if step == n_steps else HOLD_STEP)

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=55,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
