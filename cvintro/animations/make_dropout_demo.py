"""Generate dropout_demo.gif: an animated companion to the lesson 35
notebook's "Fix 3: dropout" section. A small grid of fixed activation
values stands in for one layer's output; each frame applies the notebook's
own manual_dropout() with a fresh random mask (p=0.3), zeroing a different
random subset of units and rescaling the survivors by 1/(1-p) -- making
concrete that it's a *different* random subset every single training step,
not a fixed set of "dead" units.

Run from anywhere:
    python make_dropout_demo.py
Output:
    dropout_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "dropout_demo.gif")

KEEP_COLOR = "#2ca02c"
DROP_COLOR = "#cccccc"
TEXT_COLOR = "#222222"

P = 0.3
ROWS, COLS = 5, 8
N = ROWS * COLS

act_rng = np.random.default_rng(7)
activations = act_rng.uniform(0.2, 1.0, size=N).reshape(ROWS, COLS)


def manual_dropout(x, p, rng_gen):
    keep_prob = 1 - p
    mask = (torch.rand(x.shape, generator=rng_gen) < keep_prob).float()
    return x * mask / keep_prob, mask


x_t = torch.tensor(activations, dtype=torch.float32)

N_STEPS = 5
steps = []
for step in range(N_STEPS):
    gen = torch.Generator().manual_seed(step)
    out, mask = manual_dropout(x_t, P, gen)
    steps.append((out.numpy(), mask.numpy()))


def render(step):
    out, mask = steps[step]
    frac_dropped = 1 - mask.mean()

    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=120)
    ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
    ax.grid(which="minor", color="#999999", linewidth=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_aspect("equal")

    for r in range(ROWS):
        for c in range(COLS):
            kept = mask[r, c] > 0
            face = "#eafaea" if kept else DROP_COLOR
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=face,
                                     edgecolor="white", linewidth=1.5, zorder=0))
            if kept:
                ax.text(c, r, f"{out[r, c]:.2f}", ha="center", va="center",
                         fontsize=10, color=KEEP_COLOR, fontweight="bold", zorder=1)
            else:
                ax.text(c, r, "0", ha="center", va="center", fontsize=11,
                         color="#888888", zorder=1)

    fig.suptitle(f"step {step + 1}: {frac_dropped:.0%} of units dropped this step "
                 f"(target p = {P:.0%})", fontsize=12.5, fontweight="bold",
                 color=TEXT_COLOR)
    ax.set_title("green = kept, rescaled by 1/(1-p)   |   gray = dropped this step only",
                  fontsize=10)
    fig.subplots_adjust(top=0.84, bottom=0.04)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 60
for step in range(N_STEPS):
    frames += [render(step)] * HOLD_STEP

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=55,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
