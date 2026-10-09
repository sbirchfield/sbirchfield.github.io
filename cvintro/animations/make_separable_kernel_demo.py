"""Generate separable_kernel_demo.gif: an animated walkthrough of building
the 2D blur kernel as the outer product of two 1D kernels, one cell at a
time: K[i, j] = ky[i] * kx[j]. A highlighted cell sweeps across kx while a
highlighted cell steps down ky in sync, filling the grid in reading order.

Uses the same 1D kernel as the lesson 10 notebook's classic-filters and
separable-kernels cells.

Run from anywhere:
    python make_separable_kernel_demo.py
Output:
    separable_kernel_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "separable_kernel_demo.gif")

# Same 1D kernel as the lesson 10 notebook's classic-filters/separable-kernels cells.
# Displayed as integers with the normalization factored out (1/16, 1/256), so the
# cells show clean whole numbers instead of decimals.
k1d_int = np.array([1, 4, 6, 4, 1])
K = len(k1d_int)
K_SUM = int(k1d_int.sum())  # 16

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
CELL_FACE = "#ffffff"


def fmt(v):
    return str(int(round(v)))


def draw_strip(ax, values, orientation, highlight_idx=None):
    n = len(values)
    ax.set_aspect("equal")
    ax.axis("off")
    for i, v in enumerate(values):
        is_h = orientation == "h"
        xy = (i - 0.5, -0.5) if is_h else (-0.5, i - 0.5)
        active = highlight_idx == i
        face = "#fff1e0" if active else CELL_FACE
        edge = ACTIVE_COLOR if active else "#555555"
        ax.add_patch(Rectangle(xy, 1, 1, facecolor=face, edgecolor=edge,
                                 linewidth=2.0 if active else 1.2, zorder=1))
        tx, ty = (i, 0) if is_h else (0, i)
        ax.text(tx, ty, fmt(v), ha="center", va="center", fontsize=9,
                 color=ACTIVE_COLOR if active else "#222222",
                 fontweight="bold" if active else "normal", zorder=2)
    if orientation == "h":
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, 0.5)
    else:
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(n - 0.5, -0.5)


def draw_outer_grid(ax, cells_done, highlight_cell=None):
    ax.axis("off")
    ax.set_aspect("equal")
    for i in range(K):
        for j in range(K):
            linear = i * K + j
            is_active = highlight_cell == (i, j)
            known = linear < cells_done or is_active
            face = "#fff1e0" if is_active else (CELL_FACE if known else "#f2f2f2")
            edge = ACTIVE_COLOR if is_active else "#aaaaaa"
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face,
                                     edgecolor=edge, linewidth=1.8 if is_active else 0.8, zorder=1))
            if known:
                color = ACTIVE_COLOR if is_active else "#222222"
                ax.text(j, i, fmt(k1d_int[i] * k1d_int[j]), ha="center", va="center",
                         fontsize=8.5, color=color,
                         fontweight="bold" if is_active else "normal", zorder=2)
    ax.set_xlim(-0.5, K - 0.5)
    ax.set_ylim(K - 0.5, -0.5)


def render(step, final_hold=False):
    """step: -1 for the blank intro, 0..K*K-1 for the cell currently being
    multiplied in (row = step // K, col = step % K), or K*K once the whole
    grid is filled."""
    fig = plt.figure(figsize=(8.0, 7.6), dpi=120)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.22, 1.0], height_ratios=[0.22, 1.0],
                            left=0.08, right=0.95, top=0.80, bottom=0.07,
                            wspace=0.12, hspace=0.12)
    ax_corner = fig.add_subplot(gs[0, 0])
    ax_row = fig.add_subplot(gs[0, 1])
    ax_col = fig.add_subplot(gs[1, 0])
    ax_grid = fig.add_subplot(gs[1, 1])

    ax_corner.axis("off")

    is_stepping = 0 <= step < K * K
    row, col = (step // K, step % K) if is_stepping else (None, None)

    draw_strip(ax_row, k1d_int, "h", highlight_idx=col)
    ax_row.set_title(f"kx — scale by 1/{K_SUM}", fontsize=10, color="#555555")
    draw_strip(ax_col, k1d_int, "v", highlight_idx=row)
    ax_col.set_title("ky", fontsize=10, color="#555555")

    cells_done = max(step, 0)
    draw_outer_grid(ax_grid, cells_done, highlight_cell=(row, col) if is_stepping else None)
    ax_grid.set_title(f"K = ky ⊗ kx  (outer product) — scale by 1/{K_SUM * K_SUM}", fontsize=11)

    if is_stepping:
        suptitle = f"K[{row}, {col}] = ky[{row}] × kx[{col}] = {k1d_int[row]} × {k1d_int[col]} = {k1d_int[row] * k1d_int[col]}"
        color = ACTIVE_COLOR
    elif step < 0:
        suptitle = "build the 2D blur kernel as an outer product of two 1D kernels, one cell at a time"
        color = "#222222"
    else:
        suptitle = "SAME KERNEL — the 5×5 blur kernel used earlier is exactly this outer product"
        color = MATCH_COLOR if final_hold else "#222222"

    fig.suptitle(suptitle, fontsize=12.5, color=color, fontweight="bold" if final_hold else "normal")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_INTRO = 10
HOLD_CELL = 6
HOLD_ROW_END = 12  # slightly longer pause at the end of each row
HOLD_END = 40

frames += [render(-1)] * HOLD_INTRO
for step in range(K * K):
    at_row_end = step % K == K - 1
    frames += [render(step)] * (HOLD_ROW_END if at_row_end else HOLD_CELL)
frames += [render(K * K, final_hold=True)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=110,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
