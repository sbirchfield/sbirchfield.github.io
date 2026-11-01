"""Generate mha_pipeline_static.png: a static, dimension-labeled schematic of
multi-head attention, showing the one new mechanical idea multi-head attention
adds on top of Lesson 45's single-head `attention(Q, K, V)`: split the model
dimension D into H independent heads, run attention separately within each,
concatenate the H results back together, then apply one output projection.

Reuses lesson 45's own numbers throughout -- T=197, D=768 -- rather than
inventing another arbitrary pair (lesson 45 itself never names "ViT-Base";
it just works a "base"-sized Vision Transformer example in round numbers,
which this lesson's own ViT-Base reference in lesson 47 will turn out to
match). H=12 here is ViT-Base's real head count, not an illustrative
stand-in, giving d_head = D/H = 64, exactly lesson 45's own d_k = d_v. Only
heads 1, 2, and 12 are actually drawn, with a vertical-dots placeholder
standing in for heads 3-11, so the figure stays compact without pretending
the model has fewer heads than it really does.

Each per-head attention box is annotated as being lesson 45's own diagram,
one call of it per head -- the point being that nothing new happens
*inside* a head; the new machinery is only the split beforehand and the
concat + output projection after.

Boxes are tensors (nouns); arrows are the computation connecting them
(verbs), labeled wherever that computation isn't already obvious from the
box names alone -- matching lesson 45's own convention (its SCORES-to-
SOFTMAX arrow is labeled "weights", its SOFTMAX-to-OUTPUT arrow "x V", while
plainer arrows like X-to-Q are left unlabeled). Here that means only the
final Concat-to-Output arrow needs a label ("x W_O"), since that's the one
multiplication not shown as its own box anywhere.

Run from anywhere:
    python make_mha_pipeline_static.py
Output:
    mha_pipeline_static.png  (written next to this script)
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(OUT_DIR, "mha_pipeline_static.png")

X_FACE = "#e8eef7"
HEAD_FACES = ["#fde0c4", "#d6e8d6", "#fbd6d6"]
CONCAT_FACE = "#fff3cf"
OUT_FACE = "#d0e8fa"
TEXT_COLOR = "#222222"
EDGE_COLOR = "#555555"
DIM_COLOR = "#777777"

# Lesson 45's own ViT-Base example, now with the real H=12 (not an
# illustrative stand-in) -- see module docstring.
T, D, H = 197, 768, 12
D_HEAD = D // H
SCALE = 0.095
HGT = SCALE * np.sqrt(T)
W_X = SCALE * np.sqrt(D)
W_HEAD = SCALE * np.sqrt(D_HEAD)

# Which head "slots" to actually draw, top to bottom; None = an ellipsis
# standing in for the (unshown) heads in between.
SLOTS = [1, 2, None, H]


def box(ax, x, y, w, h, label, dim, face):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                  boxstyle="round,pad=0.02,rounding_size=0.06",
                                  facecolor=face, edgecolor=EDGE_COLOR, linewidth=1.3, zorder=2))
    ax.text(x, y + h / 2 + 0.22, label, ha="center", va="bottom", fontsize=10.5,
             color=TEXT_COLOR, fontweight="bold", zorder=3)
    if dim:
        ax.text(x, y - h / 2 - 0.14, dim, ha="center", va="top", fontsize=9,
                 color=DIM_COLOR, zorder=3)


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
        ax.text(mx, my + 0.3, label, ha="center", va="center", fontsize=8.5,
                 color=color, style="italic", zorder=4)


fig, ax = plt.subplots(figsize=(13.2, 8.8), dpi=140)

GAP = 1.2
HEAD_GAP = 3.1    # vertical spacing between two real, drawn head boxes
ELLIPSIS_GAP = 1.7  # tighter spacing on either side of the vdots placeholder
x = 0.5 + W_X / 2
X = (x, 0.0)
x += W_X / 2 + GAP
heads_x = x + W_HEAD / 2
n = len(SLOTS)
# Cumulative offsets so the vdots slot sits close to its neighbors instead
# of eating a full head-sized gap on each side.
offsets = [0.0, HEAD_GAP, HEAD_GAP + ELLIPSIS_GAP, HEAD_GAP + 2 * ELLIPSIS_GAP]
mid = offsets[-1] / 2
slot_ys = [mid - o for o in offsets]
x = heads_x + W_HEAD / 2 + GAP
CONCAT = (x + W_X / 2, 0.0)
x = CONCAT[0] + W_X / 2 + GAP
OUT = (x + W_X / 2, 0.0)

box(ax, *X, W_X, HGT, "X", r"$T \times D$" + f"\n({T} x {D})", X_FACE)
first_head_pos = None
for i, (slot, y) in enumerate(zip(SLOTS, slot_ys)):
    if slot is None:
        # Center the vdots exactly between the bottom edge of the box right
        # above and the top edge of the box right below -- a font glyph's
        # own visual weight isn't reliably centered, so draw three small
        # dots by hand instead.
        above_bottom = slot_ys[i - 1] - HGT / 2
        below_top = slot_ys[i + 1] + HGT / 2
        dot_mid = (above_bottom + below_top) / 2 + 0.35  # nudge up: visual whitespace isn't exactly geometric center
        for dy in (-0.16, 0.0, 0.16):
            ax.add_patch(Circle((heads_x, dot_mid + dy), 0.028, facecolor=DIM_COLOR,
                                  edgecolor="none", zorder=3))
        continue
    # Only the first drawn head gets a dimension label underneath -- every
    # head has the identical shape, so repeating it three times is clutter.
    dim_label = r"$T \times d_{head}$" + f"\n({T} x {D_HEAD})" if slot == 1 else None
    box(ax, heads_x, y, W_HEAD, HGT, f"head {slot}\nAttention", dim_label,
        HEAD_FACES[(slot - 1) % len(HEAD_FACES)])
    arrow(ax, (X[0] + W_X / 2, X[1]), (heads_x - W_HEAD / 2, y))
    arrow(ax, (heads_x + W_HEAD / 2, y), (CONCAT[0] - W_X / 2, CONCAT[1]))
    if slot == 1:
        first_head_pos = (heads_x, y)

box(ax, *CONCAT, W_X, HGT, "Concat", r"$T \times D$" + f"\n({T} x {D})", CONCAT_FACE)
box(ax, *OUT, W_X, HGT, "Output", r"$T \times D$" + f"\n({T} x {D})", OUT_FACE)
arrow(ax, (CONCAT[0] + W_X / 2, CONCAT[1]), (OUT[0] - W_X / 2, OUT[1]), r"$\times W_O$")

split_x = X[0] + 1.0
split_y = 2.0  # below head 1's title and dim label, above X -- in the open wedge between the fan-out arrows
ax.text(split_x, split_y,
         "split D into H heads,\nattend independently in each", ha="center",
         fontsize=8.5, color=DIM_COLOR, style="italic")
concat_x = (heads_x + W_HEAD / 2) + (CONCAT[0] - (heads_x + W_HEAD / 2)) * 0.7
concat_y = -2.0  # mirror of split_y: above head H's box, below the converging arrows' center
ax.text(concat_x, concat_y,
         "concatenate the H outputs\nback into one T x D matrix", ha="center",
         fontsize=8.5, color=DIM_COLOR, style="italic")

# Plain text next to head 1's box (no pointer arrow -- its position right
# beside that box is enough to say what it refers to) tying it directly
# back to lesson 45's own diagram: every head box is that exact pipeline,
# once per head, with d_k/d_v simply renamed to d_head (every head here
# shares one query/key/value width).
note_x = first_head_pos[0] + W_HEAD / 2 + 0.35
ax.text(note_x, first_head_pos[1], "Each attention head is\nLesson 45's attention(Q, K, V),\n"
         r"with $d_k = d_v = d_{head}$" + " (renamed).",
         ha="left", va="center", fontsize=8.5, color=TEXT_COLOR, style="italic")

ax.set_xlim(-0.3, max(OUT[0] + W_X / 2 + 0.5, note_x + 3.4))
ax.set_ylim(slot_ys[-1] - HGT / 2 - 1.3, slot_ys[0] + HGT / 2 + 1.5)
ax.axis("off")
fig.suptitle("Multi-head attention: split, attend independently, concatenate, project",
             fontsize=13, color=TEXT_COLOR, fontweight="bold", y=0.99)
fig.text(0.5, 0.955, f"T = {T}, D = {D}, H = {H} (same numbers as ViT-Base in Lessons 45 and 47)",
          ha="center", fontsize=9.5, color=DIM_COLOR)
fig.subplots_adjust(left=0.01, right=0.99, top=0.89, bottom=0.05)

fig.savefig(OUT_PNG)
plt.close(fig)
print(f"Saved {OUT_PNG}")
