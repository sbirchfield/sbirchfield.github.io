"""Generate matrix_as_projections_demo.gif: an animated companion to the
lesson 30 notebook's "Matrix multiplication: many projections at once" cell.
Computes y = Wx one row at a time, using the notebook's own convolution
Toeplitz matrix (W_conv, built from shifted copies of a smoothing kernel)
and signal -- showing live that each entry of y is nothing but a single
projection (a dot product between one row of W and x), and that matrix
multiplication is simply a bank of these run one after another.

Drawn in the conventional W x = y layout: the matrix on the left, x as a
vertical column immediately to its right, an "=" sign, then y as a vertical
column -- rather than aligning x's entries over W's columns.

Reuses the notebook cell's own rng draw order (cloud consumed first, exactly
as in the notebook, before the signal draw) and W_conv construction, so the
data matches the static cell exactly.

Run from anywhere:
    python make_matrix_as_projections_demo.py
Output:
    matrix_as_projections_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "matrix_as_projections_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"

NONZERO_FILL = "#dbeafe"
ACTIVE_FILL = "#fde0c4"
DONE_FILL = "#fbe5cd"
EMPTY_FILL = "#f7f7f7"

# Same rng draw order as the notebook: `cloud` consumed first (cell 3),
# then `signal` (cell 7), so the data matches exactly.
rng = np.random.default_rng(1)
_ = rng.multivariate_normal([0, 0], [[3, 1.5], [1.5, 1]], 100)  # cloud (unused here)

signal = rng.normal(size=10)
kernel = np.array([0.25, 0.5, 0.25])

n, k = len(signal), len(kernel)
W_conv = np.zeros((n - k + 1, n))
for i in range(n - k + 1):
    W_conv[i, i:i + k] = kernel
M = W_conv.shape[0]

y_full = W_conv @ signal


def cell(ax, x0, y0, facecolor, edgecolor="#999999", linewidth=1.0, zorder=1):
    ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), 1, 1, facecolor=facecolor,
                             edgecolor=edgecolor, linewidth=linewidth, zorder=zorder))


def render(row):
    fig = plt.figure(figsize=(9.8, 6.2), dpi=120)
    gs = fig.add_gridspec(1, 4, width_ratios=[n, 1.3, 0.7, 1.3],
                            left=0.06, right=0.97, top=0.76, bottom=0.06, wspace=0.12)
    ax_W = fig.add_subplot(gs[0, 0])
    ax_x = fig.add_subplot(gs[0, 1])
    ax_eq = fig.add_subplot(gs[0, 2]); ax_eq.axis("off")
    ax_y = fig.add_subplot(gs[0, 3])

    active_cols = np.nonzero(W_conv[row])[0] if row < M else np.array([], dtype=int)

    # W: an (M x n) grid of cells, flat-colored (not value-shaded) for legibility.
    for r in range(M):
        for c in range(n):
            fc = NONZERO_FILL if W_conv[r, c] != 0 else "white"
            cell(ax_W, c, r, fc)
            if W_conv[r, c] != 0:
                ax_W.text(c, r, f"{W_conv[r, c]:.2f}", ha="center", va="center",
                           fontsize=10.5, color=NEUTRAL_COLOR)
    if row < M:
        ax_W.add_patch(Rectangle((-0.5, row - 0.5), n, 1, fill=False,
                                   edgecolor=ACTIVE_COLOR, linewidth=3, zorder=5))
    ax_W.set_xlim(-0.5, n - 0.5); ax_W.set_ylim(M - 0.5, -0.5)
    ax_W.set_aspect("equal")
    ax_W.set_xticks([]); ax_W.set_yticks(range(M))
    ax_W.set_yticklabels([f"row {r}" for r in range(M)], fontsize=10)
    ax_W.set_title("W", fontsize=14)

    # x: a vertical column, to the right of W -- the conventional Wx layout.
    for i in range(n):
        is_active = i in active_cols
        cell(ax_x, 0, i, ACTIVE_FILL if is_active else EMPTY_FILL,
             edgecolor=ACTIVE_COLOR if is_active else "#999999",
             linewidth=2.5 if is_active else 1.0, zorder=3 if is_active else 1)
        ax_x.text(0, i, f"{signal[i]:.2f}", ha="center", va="center", fontsize=10.5, color=NEUTRAL_COLOR)
    ax_x.set_xlim(-0.5, 0.5); ax_x.set_ylim(n - 0.5, -0.5)
    ax_x.set_aspect("equal")
    ax_x.set_xticks([]); ax_x.set_yticks([])
    ax_x.set_title("x", fontsize=14)

    ax_eq.text(0.5, 0.5, "=", fontsize=28, ha="center", va="center", transform=ax_eq.transAxes)

    # y: a vertical column, filled in one entry at a time as rows complete.
    for r in range(M):
        if r < row:
            cell(ax_y, 0, r, DONE_FILL)
            ax_y.text(0, r, f"{y_full[r]:.2f}", ha="center", va="center", fontsize=10.5, color=NEUTRAL_COLOR)
        elif r == row:
            cell(ax_y, 0, r, ACTIVE_FILL, edgecolor=ACTIVE_COLOR, linewidth=2.5, zorder=3)
            ax_y.text(0, r, f"{y_full[r]:.2f}", ha="center", va="center", fontsize=11,
                       color=ACTIVE_COLOR, fontweight="bold")
        else:
            cell(ax_y, 0, r, "white")
    ax_y.set_xlim(-0.5, 0.5); ax_y.set_ylim(M - 0.5, -0.5)
    ax_y.set_aspect("equal")
    ax_y.set_xticks([]); ax_y.set_yticks([])
    ax_y.set_title("y", fontsize=14)

    if row < M:
        terms = " + ".join(f"{W_conv[row, c]:.2f}·x_{{{c}}}" for c in active_cols)
        eq = f"$y_{{{row}}} = w_{{{row}}}\\cdot x = {terms} = {y_full[row]:.3f}$"
        color = ACTIVE_COLOR
    else:
        eq = f"$y = Wx$ fully computed — {M} rows, {M} projections, one matrix multiply"
        color = MATCH_COLOR
    fig.text(0.5, 0.95, "matrix multiplication = a bank of projections, one per row",
              ha="center", va="top", fontsize=14.5, color=NEUTRAL_COLOR, fontweight="bold")
    fig.text(0.5, 0.89, eq, ha="center", va="top", fontsize=12.5, color=color, family="monospace")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 22
HOLD_END = 50

for row in range(M + 1):
    frames += [render(row)] * HOLD_STEP
frames += [frames[-1]] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=90,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
