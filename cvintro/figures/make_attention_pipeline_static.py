"""Generate attention_pipeline_static.png: a static, dimension-labeled
schematic of the lesson 45 notebook's `attention(Q, K, V)` formula --
Attention(Q, K, V) = softmax(QK-transpose / sqrt(d_k)) V -- showing the shape
of every intermediate tensor so the reader can see exactly how a T x D
input becomes a T x d_v output by way of a T x T attention-weight matrix.

Box sizes are drawn roughly proportional to each tensor's actual shape
(width/height scale with sqrt(cols)/sqrt(rows), so visual *area* tracks
element count without letting the largest box swamp the figure), using the
notebook's own round-numbers example: a ViT-Base-sized setup with T=197
patches, D=768, and d_k=d_v=64. Every box in the pipeline has the same
number of rows (T), so every box ends up the same height -- only the width
changes, which is itself worth noticing: X is far wider than the T x T
attention matrix, because D=768 is much larger than T=197 here.

Q, K, V are each shown as X times its own learned weight matrix (subscript
notation: W_Q, W_K, W_V), matching the notebook's own Q = XW_Q etc. There is
no separate box for the final "x V" multiply -- the softmax weights and V
both feed directly into the Output box, since that multiply isn't itself a
tensor with a shape worth drawing a box around.

Run from anywhere:
    python make_attention_pipeline_static.py
Output:
    attention_pipeline_static.png  (written next to this script)
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(OUT_DIR, "attention_pipeline_static.png")

BOX_FACE = "#e8eef7"
Q_FACE = "#fde0c4"
K_FACE = "#d6e8d6"
V_FACE = "#e0d6f0"
OP_FACE = "#fff3cf"
OUT_FACE = "#d0e8fa"
TEXT_COLOR = "#222222"
EDGE_COLOR = "#555555"
DIM_COLOR = "#777777"

# Round-numbers ViT-Base example from the notebook text.
T, D, D_K, D_V = 197, 768, 64, 64
SCALE = 0.095
H = SCALE * np.sqrt(T)          # every box has T rows -> one shared height
W_X = SCALE * np.sqrt(D)
W_QKV = SCALE * np.sqrt(D_K)    # Q, K, V, Output all have d_k or d_v columns
W_TT = SCALE * np.sqrt(T)       # QK^T and softmax are both T x T -> square


def box(ax, x, y, w, h, label, dim, face):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                  boxstyle="round,pad=0.02,rounding_size=0.06",
                                  facecolor=face, edgecolor=EDGE_COLOR, linewidth=1.3, zorder=2))
    # Labels sit outside the box (above/below) so narrow, true-to-shape boxes
    # never have to fit text inside themselves.
    ax.text(x, y + h / 2 + 0.22, label, ha="center", va="bottom", fontsize=10.5,
             color=TEXT_COLOR, fontweight="bold", zorder=3)
    ax.text(x, y - h / 2 - 0.14, dim, ha="center", va="top", fontsize=9,
             color=DIM_COLOR, zorder=3)


# A hand-built cubic Bezier, not a matplotlib "connectionstyle" -- the
# middle two control points share p1's y and p2's y respectively, which
# forces the curve to leave p1 and arrive at p2 exactly horizontally no
# matter how far apart the two boxes sit vertically. (matplotlib's own
# angle3/angle connection styles turned out to be fragile here: angleA=0,
# angleB=180 are antiparallel lines and raise "lines do not intersect"
# whenever p1 and p2 aren't already on the same row.) When p1 and p2 share
# a y, the control points collapse onto the same line and this is just a
# straight horizontal arrow, so one helper covers both cases.
SHRINK = 0.06


def arrow(ax, p1, p2, label=None, color=EDGE_COLOR, lw=1.4):
    p1 = (p1[0] + SHRINK, p1[1])
    p2 = (p2[0] - SHRINK, p2[1])
    mid_x = (p1[0] + p2[0]) / 2
    c1, c2 = (mid_x, p1[1]), (mid_x, p2[1])
    path = Path([p1, c1, c2, p2], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=11,
                                   color=color, lw=lw, zorder=1))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.38, label, ha="center", va="center", fontsize=8.5,
                 color=color, style="italic", zorder=4)


fig, ax = plt.subplots(figsize=(13.4, 8.2), dpi=140)

GAP = 1.05
V_GAP = 2.7  # vertical spacing between Q/K/V centers -- leaves room for each
             # box's own label-above/dim-below text without overlapping its neighbor
x = 0.5 + W_X / 2
X = (x, 2.1)
x += W_X / 2 + GAP * 1.35
QKV_x = x + W_QKV / 2
Q = (QKV_x, 2.1 + V_GAP)
K = (QKV_x, 2.1)
V = (QKV_x, 2.1 - V_GAP)
x = QKV_x + W_QKV / 2 + GAP
SCORES = (x + W_TT / 2, 2.1)
x = SCORES[0] + W_TT / 2 + GAP
SOFTMAX = (x + W_TT / 2, 2.1)
x = SOFTMAX[0] + W_TT / 2 + GAP * 1.2
OUT_Y = (K[1] + V[1]) / 2
OUT = (x + W_QKV / 2, OUT_Y)

box(ax, *X, W_X, H, "X", r"$T \times D$" + f"\n({T} x {D})", BOX_FACE)
box(ax, *Q, W_QKV, H, r"$Q = XW_Q$", r"$T \times d_k$" + f"\n({T} x {D_K})", Q_FACE)
box(ax, *K, W_QKV, H, r"$K = XW_K$", r"$T \times d_k$" + f"\n({T} x {D_K})", K_FACE)
box(ax, *V, W_QKV, H, r"$V = XW_V$", r"$T \times d_v$" + f"\n({T} x {D_V})", V_FACE)
box(ax, *SCORES, W_TT, H, r"QK$^\top$ / $\sqrt{d_k}$", r"$T \times T$" + f"\n({T} x {T})", OP_FACE)
box(ax, *SOFTMAX, W_TT, H, "softmax\n(row-wise)", r"$T \times T$" + f"\n({T} x {T})", OP_FACE)
box(ax, *OUT, W_QKV, H, "Output", r"$T \times d_v$" + f"\n({T} x {D_V})", OUT_FACE)

arrow(ax, (X[0] + W_X / 2, X[1]), (Q[0] - W_QKV / 2, Q[1]))
arrow(ax, (X[0] + W_X / 2, X[1]), (K[0] - W_QKV / 2, K[1]))
arrow(ax, (X[0] + W_X / 2, X[1]), (V[0] - W_QKV / 2, V[1]))

arrow(ax, (Q[0] + W_QKV / 2, Q[1]), (SCORES[0] - W_TT / 2, SCORES[1]))
arrow(ax, (K[0] + W_QKV / 2, K[1]), (SCORES[0] - W_TT / 2, SCORES[1]))
arrow(ax, (SCORES[0] + W_TT / 2, SCORES[1]), (SOFTMAX[0] - W_TT / 2, SOFTMAX[1]), "weights")
arrow(ax, (SOFTMAX[0] + W_TT / 2, SOFTMAX[1]), (OUT[0] - W_QKV / 2, OUT[1]), "x V")
arrow(ax, (V[0] + W_QKV / 2, V[1]), (OUT[0] - W_QKV / 2, OUT[1]))

ax.text(X[0] + W_X / 2 + GAP * 0.68, Q[1] + 0.25, "three learned\nlinear projections", ha="center",
         fontsize=8.5, color=DIM_COLOR, style="italic")
ax.text(OUT[0], V[1] - H / 2 - 0.9, "each output row is a weighted\naverage of V's rows", ha="center",
         fontsize=8.5, color=DIM_COLOR, style="italic")

ax.set_xlim(-0.3, OUT[0] + W_QKV / 2 + 0.5)
ax.set_ylim(V[1] - H / 2 - 1.5, Q[1] + H / 2 + 0.7)
ax.axis("off")
fig.suptitle(r"Attention(Q, K, V) = softmax(QK$^\top$ / $\sqrt{d_k}$) V",
             fontsize=13.5, color=TEXT_COLOR, fontweight="bold", y=0.97)
fig.text(0.5, 0.925, r"$T$ = sequence length, $D$ = model dimension, $d_k$ = key/query dimension, $d_v$ = value dimension   "
          "(numbers shown: a ViT-Base-sized example)",
          ha="center", fontsize=9.5, color=DIM_COLOR)
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.05)

fig.savefig(OUT_PNG)
plt.close(fig)
print(f"Saved {OUT_PNG}")
