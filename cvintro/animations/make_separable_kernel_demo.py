"""Generate separable_kernel_demo.gif: an animated walkthrough of building
the 2D Gaussian kernel as the outer product of two 1D kernels, row by row:
K[i, j] = ky[i] * kx[j].

Uses the same 1D Gaussian kernel as the lesson 10 notebook's
separable-kernels cell.

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

# Same 1D Gaussian kernel as the lesson 10 notebook's separable-kernels cell.
sigma = 1.0
k1d = np.exp(-(np.arange(5) - 2) ** 2 / (2 * sigma ** 2))
k1d /= k1d.sum()
K = len(k1d)

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
CELL_FACE = "#ffffff"


def fmt(v):
    return "0" if abs(v) < 1e-9 else f"{v:.2f}".rstrip("0").rstrip(".")


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


def draw_outer_grid(ax, rows_done, highlight_row=None):
    ax.axis("off")
    ax.set_aspect("equal")
    for i in range(K):
        for j in range(K):
            known = i < rows_done or (highlight_row == i)
            active_row = highlight_row == i
            face = "#fff1e0" if active_row else (CELL_FACE if known else "#f2f2f2")
            edge = ACTIVE_COLOR if active_row else "#aaaaaa"
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face,
                                     edgecolor=edge, linewidth=1.6 if active_row else 0.8, zorder=1))
            if known:
                color = ACTIVE_COLOR if active_row else "#222222"
                ax.text(j, i, fmt(k1d[i] * k1d[j]), ha="center", va="center",
                         fontsize=8.5, color=color, zorder=2)
    ax.set_xlim(-0.5, K - 0.5)
    ax.set_ylim(K - 0.5, -0.5)


def render(row_i, final_hold=False):
    """row_i: -1 for the blank intro, 0..K-1 for the row currently being
    multiplied in, or K once the whole grid is filled."""
    fig = plt.figure(figsize=(8.0, 7.6), dpi=120)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.22, 1.0], height_ratios=[0.22, 1.0],
                            left=0.08, right=0.95, top=0.80, bottom=0.07,
                            wspace=0.12, hspace=0.12)
    ax_corner = fig.add_subplot(gs[0, 0])
    ax_row = fig.add_subplot(gs[0, 1])
    ax_col = fig.add_subplot(gs[1, 0])
    ax_grid = fig.add_subplot(gs[1, 1])

    ax_corner.axis("off")
    ax_corner.text(0.5, 0.5, "×", ha="center", va="center", fontsize=16, color="#888888",
                     transform=ax_corner.transAxes)

    is_stepping = 0 <= row_i < K
    hi = row_i if is_stepping else None

    draw_strip(ax_row, k1d, "h", highlight_idx=None)
    ax_row.set_title("kx (row vector)", fontsize=10, color="#555555")
    draw_strip(ax_col, k1d, "v", highlight_idx=hi)
    ax_col.set_title("ky\n(col.\nvector)", fontsize=10, color="#555555", loc="left")

    rows_done = max(row_i, 0)
    draw_outer_grid(ax_grid, rows_done, highlight_row=hi)
    ax_grid.set_title("K = ky ⊗ kx  (outer product)", fontsize=11)

    if is_stepping:
        suptitle = f"row {row_i}: K[{row_i}, j] = ky[{row_i}] × kx[j]  for every j"
        color = ACTIVE_COLOR
    elif row_i < 0:
        suptitle = "build the 2D Gaussian kernel as an outer product of two 1D kernels"
        color = "#222222"
    else:
        suptitle = "SAME KERNEL — the 5×5 Gaussian used earlier is exactly this outer product"
        color = MATCH_COLOR if final_hold else "#222222"

    fig.suptitle(suptitle, fontsize=12.5, color=color, fontweight="bold" if final_hold else "normal")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_INTRO = 10
HOLD_ROW = 16
HOLD_END = 40

frames += [render(-1)] * HOLD_INTRO
for row_i in range(K):
    frames += [render(row_i)] * HOLD_ROW
frames += [render(K, final_hold=True)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=110,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
