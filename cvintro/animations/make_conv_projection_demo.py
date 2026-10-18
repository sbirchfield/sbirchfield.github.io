"""Generate conv_projection_demo.gif: an animated companion to the lesson 34
notebook's "A conv layer is a projection, restricted and shared" section.
Slides a 3x3 edge-detecting kernel across a small single-channel image,
pausing at each position to spell out the arithmetic explicitly as
Lesson 30's projection formula: y = w^Tx + b, where x is just the current
3x3 patch flattened and w is the kernel, flattened the same way. Unlike
Lesson 10's convolution demos, the kernel is NOT flipped here -- this
matches the course's own conv2d_forward() (and PyTorch's nn.Conv2d), both
of which implement cross-correlation, not true convolution, exactly as a
conv layer actually computes it.

The image is a vertical step edge -- high values (7-9) on the left, low
values (0-2) on the right, each with a little random jitter so the numbers
aren't all identical -- so the kernel's response is still easy to predict
by eye: strongly positive right at the edge, near zero everywhere else --
foreshadowing the "Filters you already know: Lesson 12's edge detectors"
section later in the same lesson.

Run from anywhere:
    python make_conv_projection_demo.py
Output:
    conv_projection_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "conv_projection_demo.gif")

ACTIVE_COLOR = "#e07b00"
DONE_COLOR = "#3a6fb0"
NEUTRAL_COLOR = "#555555"

# A vertical step edge with a bit of jitter so the numbers aren't all
# identical, but the edge is still obvious at a glance: left half is
# randomly 7-9, right half is randomly 0-2.
rng = np.random.default_rng(2)
H0, W0 = 6, 6
image = np.where(
    np.arange(W0)[None, :] < W0 // 2,
    rng.integers(6, 10, size=(H0, W0)),
    rng.integers(0, 4, size=(H0, W0)),
).astype(np.float64)

# Lesson 12's vertical-edge Prewitt kernel: no flip, no padding -- a direct
# cross-correlation, matching conv2d_forward()'s own `patch * kernel`.
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1],
], dtype=np.float64)
bias = 0.0

H, W = image.shape
K = kernel.shape[0]
OH, OW = H - K + 1, W - K + 1


def conv_at(r, c):
    patch = image[r:r + K, c:c + K]
    return float(np.sum(patch * kernel) + bias)


full_output = np.array([[conv_at(r, c) for c in range(OW)] for r in range(OH)])


def draw_small_grid(ax, M, title, border_color, cell_text_color="#222222"):
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


def draw_math_panel(ax, r, c):
    ax.axis("off")
    patch = image[r:r + K, c:c + K]
    ax_p = ax.inset_axes([0.05, 0.56, 0.9, 0.38])
    ax_k = ax.inset_axes([0.05, 0.04, 0.9, 0.38])
    draw_small_grid(ax_p, patch, "patch x", ACTIVE_COLOR, cell_text_color=ACTIVE_COLOR)
    draw_small_grid(ax_k, kernel, "kernel w", NEUTRAL_COLOR)
    val = full_output[r, c]
    ax.text(0.5, -0.04, f"y = wᵀx + b = {val:g}", ha="center",
             fontsize=12, fontweight="bold", color=ACTIVE_COLOR, transform=ax.transAxes)


def draw_image_grid(ax, highlight_rc=None):
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="#aaaaaa", linewidth=0.7)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.tick_params(length=0, labelsize=7)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)

    for rr in range(H):
        for cc in range(W):
            ax.add_patch(Rectangle((cc - 0.5, rr - 0.5), 1, 1, facecolor="white",
                                     edgecolor="none", zorder=0))
            ax.text(cc, rr, f"{image[rr, cc]:g}", ha="center", va="center",
                     fontsize=9, color="#222222", zorder=1)

    if highlight_rc is not None:
        r, c = highlight_rc
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), K, K, facecolor=ACTIVE_COLOR,
                                 alpha=0.22, edgecolor=ACTIVE_COLOR, linewidth=2.6, zorder=3))

    ax.set_title("input image (one channel)", fontsize=12)


def draw_output_grid(ax, computed_mask, current_rc=None):
    for r in range(OH):
        for c in range(OW):
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

    ax.set_xticks(range(OW))
    ax.set_yticks(range(OH))
    ax.tick_params(length=0, labelsize=7)
    ax.set_xlim(-0.5, OW - 0.5)
    ax.set_ylim(OH - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    ax.set_title("output feature map", fontsize=12)


def render(step):
    """step: a flat index 0..OH*OW-1 for the position currently being
    computed (row-major order), or OH*OW once every position is done."""
    total = OH * OW
    is_stepping = step < total
    r, c = divmod(min(step, total - 1), OW)

    fig, (ax_math, ax_img, ax_out) = plt.subplots(
        1, 3, figsize=(13.2, 5.0), dpi=120,
        gridspec_kw={"width_ratios": [0.55, 1.0, 0.75]},
    )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.78, bottom=0.1, wspace=0.3)

    draw_math_panel(ax_math, r, c)
    draw_image_grid(ax_img, highlight_rc=(r, c) if is_stepping else None)

    computed_mask = np.zeros((OH, OW), dtype=bool)
    flat = min(step, total)
    for k in range(flat):
        rr, cc = divmod(k, OW)
        computed_mask[rr, cc] = True
    draw_output_grid(ax_out, computed_mask, current_rc=(r, c) if is_stepping else None)

    if is_stepping:
        title = f"convolution with a kernel is wᵀx + b, with b={bias:g}"
        color = ACTIVE_COLOR
    else:
        title = "one 3x3 kernel, reused everywhere"
        color = DONE_COLOR
    fig.suptitle(title, fontsize=12.5, color=color, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 45
HOLD_END = 90
total = OH * OW

for step in range(total + 1):
    frames += [render(step)] * (HOLD_END if step == total else HOLD_STEP)

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=55,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
