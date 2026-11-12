"""Generate encoder_block_static.png: the classic Transformer encoder-block
diagram (Vaswani et al., 2017), matching the lesson 46 notebook's own
`EncoderBlock`: multi-head attention and a per-position feedforward network,
each wrapped in its own residual connection (Lesson 37) and layer norm
(normalizing across the feature dimension per token, not across the batch).

Drawn as a single vertical pipeline -- input at the top, output at the
bottom, reading top to bottom like the course's other static diagrams
(e.g. lesson 47's vit_architecture_diagram.png) -- with the two residual
connections shown as curved arrows that skip around their sub-layer and
rejoin at the "+" just before each LayerNorm.

Run from anywhere:
    python make_encoder_block_static.py
Output:
    encoder_block_static.png  (written next to this script)
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(OUT_DIR, "encoder_block_static.png")

IO_FACE = "#d0e8fa"
MHA_FACE = "#fde0c4"
FF_FACE = "#e0d6f0"
NORM_FACE = "#fff3cf"
SUM_FACE = "#ffffff"
TEXT_COLOR = "#222222"
EDGE_COLOR = "#555555"
RESID_COLOR = "#2e7d32"
DIM_COLOR = "#777777"

W_MAIN = 2.6
H_MAIN = 0.62
W_SUM = 0.36


def box(ax, x, y, w, h, label, face, fontsize=11):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                  boxstyle="round,pad=0.02,rounding_size=0.08",
                                  facecolor=face, edgecolor=EDGE_COLOR, linewidth=1.3, zorder=2))
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
             color=TEXT_COLOR, fontweight="bold", zorder=3)


def sum_node(ax, x, y):
    ax.add_patch(Circle((x, y), W_SUM / 2, facecolor=SUM_FACE, edgecolor=EDGE_COLOR,
                          linewidth=1.3, zorder=2))
    ax.text(x, y, "+", ha="center", va="center", fontsize=14, fontweight="bold",
             color=TEXT_COLOR, zorder=3)


def vline(ax, x, y1, y2, color=EDGE_COLOR, lw=1.6):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", mutation_scale=12,
                                   color=color, lw=lw, zorder=1))


def residual_skip(ax, x_main, y1, y2, bulge):
    """A curved arrow leaving the main spine at y1, bulging out to the side,
    and rejoining the spine at y2 -- the residual connection that skips over
    whatever sub-layer sits between y1 and y2."""
    x_side = x_main + bulge
    p1 = (x_main + 0.02, y1)
    p2 = (x_main + 0.02, y2)
    c1 = (x_side, y1)
    c2 = (x_side, y2)
    path = Path([p1, c1, c2, p2], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=12,
                                   color=RESID_COLOR, lw=1.6, zorder=1,
                                   linestyle=(0, (5, 2))))


fig, ax = plt.subplots(figsize=(6.6, 10.2), dpi=140)

x = 0.0
y_in = 0.0
y_mha = y_in + 1.3
y_sum1 = y_mha + 1.0
y_norm1 = y_sum1 + 0.75
y_ff = y_norm1 + 1.3
y_sum2 = y_ff + 1.0
y_norm2 = y_sum2 + 0.75
y_out = y_norm2 + 1.0

box(ax, x, y_in, W_MAIN, H_MAIN, "Input  (T x D)", IO_FACE)
box(ax, x, y_mha, W_MAIN, H_MAIN * 1.3, "Multi-Head\nAttention", MHA_FACE, fontsize=10.5)
sum_node(ax, x, y_sum1)
box(ax, x, y_norm1, W_MAIN, H_MAIN, "LayerNorm", NORM_FACE)
box(ax, x, y_ff, W_MAIN, H_MAIN * 1.3, "Feedforward\n(Linear, GELU, Linear)", FF_FACE, fontsize=10.5)
sum_node(ax, x, y_sum2)
box(ax, x, y_norm2, W_MAIN, H_MAIN, "LayerNorm", NORM_FACE)
box(ax, x, y_out, W_MAIN, H_MAIN, "Output  (T x D)", IO_FACE)

vline(ax, x, y_in + H_MAIN / 2, y_mha - H_MAIN * 1.3 / 2)
vline(ax, x, y_mha + H_MAIN * 1.3 / 2, y_sum1 - W_SUM / 2)
vline(ax, x, y_sum1 + W_SUM / 2, y_norm1 - H_MAIN / 2)
vline(ax, x, y_norm1 + H_MAIN / 2, y_ff - H_MAIN * 1.3 / 2)
vline(ax, x, y_ff + H_MAIN * 1.3 / 2, y_sum2 - W_SUM / 2)
vline(ax, x, y_sum2 + W_SUM / 2, y_norm2 - H_MAIN / 2)
vline(ax, x, y_norm2 + H_MAIN / 2, y_out - H_MAIN / 2)

residual_skip(ax, x + W_MAIN / 2, y_in + H_MAIN / 2, y_sum1, bulge=1.05)
residual_skip(ax, x + W_MAIN / 2, y_norm1 + H_MAIN / 2, y_sum2, bulge=1.05)

ax.text(x + W_MAIN / 2 + 1.25, (y_in + H_MAIN / 2 + y_sum1) / 2, "residual", ha="center",
         fontsize=9, color=RESID_COLOR, style="italic", rotation=90)
ax.text(x + W_MAIN / 2 + 1.25, (y_norm1 + H_MAIN / 2 + y_sum2) / 2, "residual", ha="center",
         fontsize=9, color=RESID_COLOR, style="italic", rotation=90)

ax.set_xlim(x - W_MAIN / 2 - 0.6, x + W_MAIN / 2 + 2.1)
ax.set_ylim(y_in - 0.6, y_out + 0.6)
ax.invert_yaxis()  # input at top, output at bottom -- read top to bottom
ax.axis("off")
fig.suptitle("The Transformer encoder block", fontsize=14, color=TEXT_COLOR,
             fontweight="bold", y=0.975)
fig.text(0.5, 0.945, "every sub-layer is wrapped in a residual connection and a LayerNorm",
          ha="center", fontsize=9.5, color=DIM_COLOR)
fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.03)

fig.savefig(OUT_PNG)
plt.close(fig)
print(f"Saved {OUT_PNG}")
