"""Generate batchnorm_vs_layernorm_static.png: a side-by-side diagram
answering the one question that actually distinguishes BatchNorm (Lesson 37)
from LayerNorm (Lesson 46): which axis gets averaged over?

Both grids use the same column axis -- channels, i.e. features, the same
thing under two names depending on whether you're in CNN or Transformer
vocabulary -- so the *only* thing that visibly differs between the two
panels is the row axis, which is the actual point of the diagram:
  - BatchNorm: rows = different, unrelated examples in a mini-batch (e.g.
    8 different photos). Normalizes down each COLUMN (one channel's
    statistics, pooled across every example in the batch).
  - LayerNorm: rows = different positions within ONE example's own
    sequence (e.g. one sentence's words, or one image's patches -- this
    grid shows a single example from the batch, not the whole batch).
    Normalizes across each ROW (one token's own channels, independent of
    every other token and of every other example in the batch).

Run from anywhere:
    python make_batchnorm_vs_layernorm_static.py
Output:
    batchnorm_vs_layernorm_static.png  (written next to this script)
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(OUT_DIR, "batchnorm_vs_layernorm_static.png")

CELL_FACE = "#eef1f5"
HIGHLIGHT_FACE = "#fde0c4"
HIGHLIGHT_EDGE = "#e07b00"
EDGE_COLOR = "#999999"
TEXT_COLOR = "#222222"
DIM_COLOR = "#777777"

ROWS, COLS = 6, 5
CELL = 0.5
GAP = 0.06


def draw_grid(ax, highlight_row=None, highlight_col=None):
    for r in range(ROWS):
        for c in range(COLS):
            x, y = c * (CELL + GAP), (ROWS - 1 - r) * (CELL + GAP)
            face = CELL_FACE
            ax.add_patch(Rectangle((x, y), CELL, CELL, facecolor=face,
                                     edgecolor=EDGE_COLOR, linewidth=0.8, zorder=2))
    if highlight_col is not None:
        x = highlight_col * (CELL + GAP) - GAP / 2
        ax.add_patch(FancyBboxPatch((x, -GAP / 2), CELL + GAP, ROWS * (CELL + GAP) - GAP / 2,
                                      boxstyle="round,pad=0.015,rounding_size=0.04",
                                      facecolor=HIGHLIGHT_FACE, edgecolor=HIGHLIGHT_EDGE,
                                      linewidth=2.2, zorder=1, alpha=0.85))
    if highlight_row is not None:
        y = (ROWS - 1 - highlight_row) * (CELL + GAP) - GAP / 2
        ax.add_patch(FancyBboxPatch((-GAP / 2, y), COLS * (CELL + GAP) - GAP / 2, CELL + GAP,
                                      boxstyle="round,pad=0.015,rounding_size=0.04",
                                      facecolor=HIGHLIGHT_FACE, edgecolor=HIGHLIGHT_EDGE,
                                      linewidth=2.2, zorder=1, alpha=0.85))
    ax.set_xlim(-0.5, COLS * (CELL + GAP) + 0.1)
    ax.set_ylim(-0.6, ROWS * (CELL + GAP) + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")


fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.6), dpi=140)

ax = axes[0]
draw_grid(ax, highlight_col=2)
ax.text(COLS * (CELL + GAP) / 2 - GAP / 2, ROWS * (CELL + GAP) + 0.35, "BatchNorm", ha="center",
         fontsize=13, color=TEXT_COLOR, fontweight="bold")
ax.text(-0.35, ROWS * (CELL + GAP) / 2 - GAP / 2, "examples in the batch", ha="center", va="center",
         fontsize=9.5, color=DIM_COLOR, rotation=90)
ax.text(COLS * (CELL + GAP) / 2 - GAP / 2, ROWS * (CELL + GAP) + 0.12, "channels (features)", ha="center",
         fontsize=9.5, color=DIM_COLOR)
ax.text(COLS * (CELL + GAP) / 2 - GAP / 2, -0.45,
         "normalize down each column:\none channel's mean/std, pooled across the batch",
         ha="center", fontsize=9, color=HIGHLIGHT_EDGE, style="italic")

ax = axes[1]
draw_grid(ax, highlight_row=2)
ax.text(COLS * (CELL + GAP) / 2 - GAP / 2, ROWS * (CELL + GAP) + 0.35, "LayerNorm", ha="center",
         fontsize=13, color=TEXT_COLOR, fontweight="bold")
ax.text(-0.35, ROWS * (CELL + GAP) / 2 - GAP / 2, "tokens in one example", ha="center", va="center",
         fontsize=9.5, color=DIM_COLOR, rotation=90)
ax.text(COLS * (CELL + GAP) / 2 - GAP / 2, ROWS * (CELL + GAP) + 0.12, "channels (features)", ha="center",
         fontsize=9.5, color=DIM_COLOR)
ax.text(COLS * (CELL + GAP) / 2 - GAP / 2, -0.45,
         "normalize across each row:\none token's own mean/std, independent of every other token",
         ha="center", fontsize=9, color=HIGHLIGHT_EDGE, style="italic")
ax.text(COLS * (CELL + GAP) / 2 - GAP / 2, ROWS * (CELL + GAP) + 0.24,
         "(one example from the batch)", ha="center", fontsize=8.5, color=DIM_COLOR, style="italic")

fig.suptitle("BatchNorm vs. LayerNorm: which axis gets averaged over?",
             fontsize=14, color=TEXT_COLOR, fontweight="bold", y=0.99)
fig.subplots_adjust(left=0.06, right=0.97, top=0.86, bottom=0.08, wspace=0.35)

fig.savefig(OUT_PNG)
plt.close(fig)
print(f"Saved {OUT_PNG}")
