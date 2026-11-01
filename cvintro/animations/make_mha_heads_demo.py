"""Generate mha_heads_demo.gif: an animated companion to the lesson 46
notebook's multi-head attention section, illustrating the claim that
different heads specialize in different kinds of relationships. Shows three
heads' full attention-weight matrices for the same small token sequence side
by side, then steps through query positions one at a time, highlighting that
row in all three heads at once so the reader can directly compare what each
head focuses on for the same query.

The three heads' weights are hand-constructed (not learned) so each one
shows a clean, recognizable pattern instead of the noisy mess an untrained
or even a real trained head often looks like at a glance:
  - head 1: attends mostly to the immediately preceding token (local).
  - head 2: attends mostly to a single fixed "anchor" position (position 0).
  - head 3: spreads attention broadly across the whole sequence (global).
This is illustrative, not a claim about what real trained heads learn --
the lesson text is careful to say heads merely *can* specialize this way.

Run from anywhere:
    python make_mha_heads_demo.py
Output:
    mha_heads_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "mha_heads_demo.gif")

ACTIVE_COLOR = "#e07b00"
NEUTRAL_COLOR = "#222222"

TOKENS = ["the", "cat", "sat", "on", "the", "mat"]
T = len(TOKENS)
HEAD_NAMES = ["head 1: local (previous token)", "head 2: fixed anchor (position 0)", "head 3: global (spread out)"]

logits = np.zeros((3, T, T))
for i in range(T):
    for j in range(T):
        # head 1: local -- strong peak at j == i-1 (or self, at i == 0)
        target = max(i - 1, 0)
        logits[0, i, j] = -3.0 * abs(j - target)
        # head 2: anchor -- always prefers position 0, mildly prefers self too
        logits[1, i, j] = 3.0 if j == 0 else (1.0 if j == i else 0.0)
        # head 3: global -- nearly flat, tiny self-bias only
        logits[2, i, j] = 0.3 if j == i else 0.0

weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
weights /= weights.sum(axis=-1, keepdims=True)


def render(i):
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.6), dpi=120)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.72, bottom=0.14, wspace=0.35)

    for h, ax in enumerate(axes):
        ax.imshow(weights[h], cmap="viridis", vmin=0, vmax=weights[h].max(), aspect="equal")
        ax.set_xticks(range(T)); ax.set_xticklabels(TOKENS, fontsize=7.5, rotation=45, ha="right")
        ax.set_yticks(range(T)); ax.set_yticklabels(TOKENS, fontsize=7.5)
        ax.add_patch(Rectangle((-0.5, i - 0.5), T, 1, fill=False, edgecolor=ACTIVE_COLOR, linewidth=2.6, zorder=5))
        ax.set_title(HEAD_NAMES[h], fontsize=9.5)

    fig.suptitle(f"query position {i} (\"{TOKENS[i]}\"): each head attends very differently",
                 fontsize=12.5, color=NEUTRAL_COLOR, fontweight="bold", y=0.96)
    fig.text(0.5, 0.86, "same input sequence, same query position -- three different attention patterns",
              ha="center", fontsize=9.5, color="#666666", style="italic")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 55
HOLD_END = 100
for i in range(T):
    frames += [render(i)] * (HOLD_END if i == T - 1 else HOLD_STEP)

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=60,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
