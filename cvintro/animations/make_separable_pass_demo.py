"""Generate separable_pass_demo.gif: an animated walkthrough of applying a
separable kernel as two 1D passes -- convolve each row with kx, then
convolve each column of that with ky -- compared against convolving with
the full 2D kernel directly.

Uses the same image and kernel values as the lesson 10 notebook's
classic-filters and separable-kernels cells.

Run from anywhere:
    python make_separable_pass_demo.py
Output:
    separable_pass_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "separable_pass_demo.gif")

# Same 1D kernel as the separable_kernel_demo.gif / lesson 10 notebook's
# classic-filters/separable-kernels cells: [1, 4, 6, 4, 1], scale by 1/16.
# Kept as plain integers here too -- convolving integer image with an integer
# kernel stays exactly integer, so every panel displays clean whole numbers
# with the normalization factored out as a "scale by 1/N" label underneath.
k1d_int = np.array([1., 4., 6., 4., 1.])
K = len(k1d_int)
K_SUM = int(k1d_int.sum())  # 16
PAD = K // 2
gaussian2d_int = np.outer(k1d_int, k1d_int)  # scale by 1/(16*16) = 1/256

# Same small "bump" image used in the other 2D convolution demos.
image = np.array([
    [1, 2, 3, 2, 1],
    [2, 3, 4, 3, 2],
    [3, 4, 5, 4, 3],
    [2, 3, 4, 3, 2],
    [1, 2, 3, 2, 1],
], dtype=np.float64)
H, W = image.shape


def conv1d(signal, kernel):
    pad = len(kernel) // 2
    padded = np.pad(signal, pad, mode="constant")
    flipped = kernel[::-1]  # symmetric here, but keep the real definition
    out = np.zeros_like(signal, dtype=np.float64)
    for n in range(len(signal)):
        out[n] = np.dot(padded[n:n + len(kernel)], flipped)
    return out


row_pass = np.array([conv1d(image[r, :], k1d_int) for r in range(H)])             # convolve each row
final_separable = np.array([conv1d(row_pass[:, c], k1d_int) for c in range(W)]).T  # then each column

padded_img = np.pad(image, PAD, mode="constant")
flipped2d = gaussian2d_int[::-1, ::-1]
direct_2d = np.array([
    [float(np.sum(padded_img[r:r + K, c:c + K] * flipped2d)) for c in range(W)]
    for r in range(H)
])

assert np.allclose(final_separable, direct_2d), "separable result does not match direct 2D convolution!"

ACTIVE_COLOR = "#e07b00"
DONE_COLOR = "#3a6fb0"
MATCH_COLOR = "#2e9e44"
CELL_FACE = "#ffffff"


def fmt(v):
    return str(int(round(v)))


def draw_image_panel(ax, M, title, face=CELL_FACE, text_color="#222222", show=True, scale_label=None):
    for r in range(H):
        for c in range(W):
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=face if show else "#f2f2f2",
                                     edgecolor="#aaaaaa", linewidth=0.7, zorder=0))
            if show:
                ax.text(c, r, fmt(M[r, c]), ha="center", va="center", fontsize=9,
                         color=text_color, fontweight="bold", zorder=1)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    if show and scale_label:
        ax.text(0.5, -0.08, scale_label, transform=ax.transAxes, ha="center", va="top",
                 fontsize=9, color="#555555")


def draw_progressive_panel(ax, M, title, axis, idx, total, scale_label=None):
    """Reveal M one row (axis='row') or one column (axis='col') at a time.
    idx: None = nothing shown yet; 0..total-1 = that row/col is the current
    highlight (earlier ones done, later ones hidden); total = fully done."""
    for r in range(H):
        for c in range(W):
            line = r if axis == "row" else c
            if idx is None:
                state = "hidden"
            elif idx >= total or line < idx:
                state = "done"
            elif line == idx:
                state = "current"
            else:
                state = "hidden"
            face_cell = {"hidden": "#f2f2f2", "current": "#fff1e0", "done": CELL_FACE}[state]
            edge_cell = ACTIVE_COLOR if state == "current" else "#aaaaaa"
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=face_cell,
                                     edgecolor=edge_cell, linewidth=1.6 if state == "current" else 0.7, zorder=0))
            if state != "hidden":
                text_color = ACTIVE_COLOR if state == "current" else DONE_COLOR
                ax.text(c, r, fmt(M[r, c]), ha="center", va="center", fontsize=9,
                         color=text_color, fontweight="bold", zorder=1)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    if idx is not None and scale_label:
        ax.text(0.5, -0.08, scale_label, transform=ax.transAxes, ha="center", va="top",
                 fontsize=9, color="#555555")


def render(row_idx=None, col_idx=None, final=False):
    """row_idx/col_idx: None, 0..total-1 (stepping), or total (fully done).
    Column stepping only happens once rows are fully done."""
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.8), dpi=120,
                               gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    fig.subplots_adjust(left=0.03, right=0.98, top=0.72, bottom=0.14, wspace=0.35)
    ax_img, ax_mid, ax_final, ax_direct = axes

    draw_image_panel(ax_img, image, "image")
    draw_progressive_panel(ax_mid, row_pass, "after row pass\n(⊛ kx)", "row", row_idx, H,
                             scale_label=f"scale by 1/{K_SUM}")
    draw_progressive_panel(ax_final, final_separable, "after column pass\n(⊛ ky)", "col", col_idx, W,
                             scale_label=f"scale by 1/{K_SUM * K_SUM}")
    draw_image_panel(ax_direct, direct_2d, "direct 2D conv.\n(⊛ K)",
                       face="#eafaf0" if final else CELL_FACE,
                       text_color=MATCH_COLOR if final else "#222222", show=final,
                       scale_label=f"scale by 1/{K_SUM * K_SUM}")

    if final:
        suptitle = "compare to convolving with the full 5×5 kernel directly — identical"
        color = MATCH_COLOR
    elif col_idx is not None and 0 <= col_idx < W:
        suptitle = f"pass 2: convolve column {col_idx} of the row-pass result with ky"
        color = ACTIVE_COLOR
    elif col_idx == W:
        suptitle = "pass 2 complete: every column convolved with ky"
        color = "#222222"
    elif row_idx is not None and 0 <= row_idx < H:
        suptitle = f"pass 1: convolve row {row_idx} of the image with kx"
        color = ACTIVE_COLOR
    elif row_idx == H:
        suptitle = "pass 1 complete: every row convolved with kx"
        color = "#222222"
    else:
        suptitle = "start with the image"
        color = "#222222"
    fig.suptitle(suptitle, fontsize=13, color=color)

    if final:
        ax_final.text(0.5, 1.26, "=", transform=ax_final.transAxes, ha="center", va="bottom",
                        fontsize=18, color=MATCH_COLOR, fontweight="bold")
        ax_direct.text(0.5, 1.26, "MATCH", transform=ax_direct.transAxes, ha="center", va="bottom",
                         fontsize=14, color=MATCH_COLOR, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_INTRO = 12
HOLD_LINE = 10
HOLD_PHASE_DONE = 14
HOLD_END = 40

frames += [render()] * HOLD_INTRO                                   # start: image only
for r in range(H):
    frames += [render(row_idx=r)] * HOLD_LINE                       # pass 1, row by row
frames += [render(row_idx=H)] * HOLD_PHASE_DONE                     # pass 1 complete
for c in range(W):
    frames += [render(row_idx=H, col_idx=c)] * HOLD_LINE            # pass 2, column by column
frames += [render(row_idx=H, col_idx=W)] * HOLD_PHASE_DONE          # pass 2 complete
frames += [render(row_idx=H, col_idx=W, final=True)] * HOLD_END     # final: compare to direct 2D

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=110,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
