"""Generate convolution2d_demo.gif: an animated, step-by-step walkthrough of
2D convolution "the traditional way" -- flip the kernel (180 degrees), slide
it across the zero-padded image, and at each position take the dot product
with the overlapping window. This is the direct, position-by-position view
of convolution; the notebook's `convolve2d` implements the same math but
loops over the kernel's few entries instead, for speed.

Uses the same asymmetric kernel as the lesson 10 notebook's
convolution-vs-correlation example, so the "flip actually matters" point
carries over.

Run from anywhere:
    python make_convolution2d_demo.py
Output:
    convolution2d_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "convolution2d_demo.gif")

# A small "bump" image (the 2D analog of the 1D ramp signal used earlier),
# and the same asymmetric kernel used later in the notebook's
# convolution-vs-correlation comparison.
image = np.array([
    [1, 2, 3, 2, 1],
    [2, 3, 4, 3, 2],
    [3, 4, 5, 4, 3],
    [2, 3, 4, 3, 2],
    [1, 2, 3, 2, 1],
], dtype=np.float64)

kernel = np.array([[1, 2, -1],
                    [0, 1, 3],
                    [-2, 1, 0]], dtype=np.float64)
flipped = kernel[::-1, ::-1]  # 180-degree rotation: what actually slides over the image

H, W = image.shape
KSIZE = kernel.shape[0]
PAD = KSIZE // 2
padded = np.pad(image, PAD, mode="constant")

ACTIVE_COLOR = "#e07b00"
DONE_COLOR = "#3a6fb0"
PAD_FACE = "#dddddd"
IMG_FACE = "#ffffff"


def conv_at(r, c):
    window = padded[r:r + KSIZE, c:c + KSIZE]
    return float(np.sum(window * flipped))


full_output = np.array([[conv_at(r, c) for c in range(W)] for r in range(H)])


def draw_small_grid(ax, M, title, border_color, cell_text_color="#222222"):
    """A static kxk panel showing a kernel's numeric entries."""
    k = M.shape[0]
    ax.imshow(np.ones((k, k, 3)), interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which="minor", color="#888888", linewidth=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for r in range(k):
        for c in range(k):
            ax.text(c, r, f"{M[r, c]:g}", ha="center", va="center",
                     fontsize=10, color=cell_text_color)
    ax.set_xlim(-0.5, k - 0.5)
    ax.set_ylim(k - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    ax.set_title(title, fontsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(2.2)


def draw_kernel_panel(ax):
    ax.axis("off")
    ax_k = ax.inset_axes([0.05, 0.56, 0.9, 0.38])
    ax_f = ax.inset_axes([0.05, 0.04, 0.9, 0.38])
    draw_small_grid(ax_k, kernel, "kernel K", "#555555")
    draw_small_grid(ax_f, flipped, "flipped (180°)", ACTIVE_COLOR, cell_text_color=ACTIVE_COLOR)
    ax.annotate("", xy=(0.5, 0.49), xytext=(0.5, 0.555),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.6),
                xycoords="axes fraction")
    ax.text(0.38, 0.5225, "flip", ha="right", va="center", fontsize=11,
             color="#555555", style="italic", transform=ax.transAxes)


def draw_image_grid(ax, highlight_rc=None):
    Hp, Wp = padded.shape
    ax.set_xticks(np.arange(-0.5, Wp, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, Wp, 1), minor=True)
    ax.grid(which="minor", color="#aaaaaa", linewidth=0.7)
    ax.set_xticks(range(Wp))
    ax.set_yticks(range(Hp))
    ax.tick_params(length=0, labelsize=7)
    ax.set_xlim(-0.5, Wp - 0.5)
    ax.set_ylim(Hp - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)

    for rr in range(Hp):
        for cc in range(Wp):
            is_pad = rr < PAD or rr >= PAD + H or cc < PAD or cc >= PAD + W
            face = PAD_FACE if is_pad else IMG_FACE
            ax.add_patch(Rectangle((cc - 0.5, rr - 0.5), 1, 1, facecolor=face,
                                     edgecolor="none", zorder=0))
            if not is_pad:
                ax.text(cc, rr, f"{padded[rr, cc]:g}", ha="center", va="center",
                         fontsize=9, color="#222222", zorder=1)

    if highlight_rc is not None:
        r, c = highlight_rc
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), KSIZE, KSIZE, fill=False,
                                 edgecolor=ACTIVE_COLOR, linewidth=2.6, zorder=3))
        for i in range(KSIZE):
            for j in range(KSIZE):
                ax.text(c + j - 0.32, r + i - 0.32, f"{flipped[i, j]:g}",
                         ha="left", va="top", fontsize=7.5, color=ACTIVE_COLOR,
                         fontweight="bold", zorder=4)

    ax.set_title("image (gray = zero padding)", fontsize=12)


def draw_output_grid(ax, computed_mask, current_rc=None):
    for r in range(H):
        for c in range(W):
            if current_rc == (r, c):
                face, text_color, val = "#fff1e0", ACTIVE_COLOR, full_output[r, c]
                show = True
            elif computed_mask[r, c]:
                face, text_color, val = "#eef3fa", DONE_COLOR, full_output[r, c]
                show = True
            else:
                face, text_color, val = "#ffffff", None, None
                show = False
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=face,
                                     edgecolor="#aaaaaa", linewidth=0.7, zorder=0))
            if show:
                ax.text(c, r, f"{val:g}", ha="center", va="center", fontsize=9,
                         color=text_color, fontweight="bold", zorder=1)
            if current_rc == (r, c):
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                         edgecolor=ACTIVE_COLOR, linewidth=2.6, zorder=2))

    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.tick_params(length=0, labelsize=7)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    ax.set_title("output = image * kernel", fontsize=12)


def render(step, final_hold=False):
    """step: -1 for the blank intro, a flat index 0..H*W-1 for the position
    currently being computed (row-major order), or H*W once every position
    is done."""
    total = H * W
    is_stepping = 0 <= step < total

    fig, (ax_kernel, ax_img, ax_out) = plt.subplots(
        1, 3, figsize=(14.5, 5.4), dpi=120,
        gridspec_kw={"width_ratios": [0.5, 1.0, 1.0]},
    )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.8, bottom=0.06, wspace=0.28)

    draw_kernel_panel(ax_kernel)

    computed_mask = np.zeros((H, W), dtype=bool)
    if step >= 0:
        flat = min(step, total)
        computed_mask.flat[:flat] = True

    current_rc = None
    if is_stepping:
        r, c = divmod(step, W)
        current_rc = (r, c)
        draw_image_grid(ax_img, highlight_rc=(r, c))
    else:
        draw_image_grid(ax_img)

    draw_output_grid(ax_out, computed_mask, current_rc=current_rc)

    if is_stepping:
        r, c = current_rc
        suptitle = f"position (row={r}, col={c}): output = {full_output[r, c]:g}"
        color = ACTIVE_COLOR
    elif step < 0:
        suptitle = "flipped kernel will slide across the zero-padded image"
        color = "#222222"
    else:
        suptitle = ""
        color = "#222222"

    if suptitle:
        fig.suptitle(suptitle, fontsize=12.5, color=color)

    if not is_stepping and step >= 0:
        ax_out.text(0.5, 1.18, "FINAL RESULT", transform=ax_out.transAxes,
                     ha="center", va="bottom", fontsize=15, fontweight="bold",
                     color="#1f1f1f")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_INTRO = 10
HOLD_STEP = 7
HOLD_END = 28

frames += [render(-1)] * HOLD_INTRO

for step in range(H * W):
    frames += [render(step)] * HOLD_STEP

frames += [render(H * W, final_hold=True)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=100,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames, {H * W} positions)")
