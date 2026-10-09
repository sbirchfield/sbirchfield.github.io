"""Generate convolution2d_fast_demo.gif: an animated walkthrough of the
"efficient trick" version of 2D convolution used by the notebook's
convolve2d() -- instead of sliding a window and computing one dot product
per output pixel, loop over the kernel's (few) entries, and for each one add
a shifted, scaled copy of the *whole* padded image to an accumulator. Same
image and kernel as convolution2d_demo.gif (the position-by-position
version), so the two are a direct contrast: there, one output cell lights up
per step; here, the entire accumulator updates at once per step.

Run from anywhere:
    python make_convolution2d_fast_demo.py
Output:
    convolution2d_fast_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "convolution2d_fast_demo.gif")

# Same image and kernel as convolution2d_demo.gif.
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
flipped = kernel[::-1, ::-1]

H, W = image.shape
KSIZE = kernel.shape[0]
PAD = KSIZE // 2
padded = np.pad(image, PAD, mode="constant")

# Reference result, computed the position-by-position way (same definition
# as convolution2d_demo.gif), to sanity-check the accumulator trick below.
reference_output = np.array([
    [float(np.sum(padded[r:r + KSIZE, c:c + KSIZE] * flipped)) for c in range(W)]
    for r in range(H)
])

ACTIVE_COLOR = "#e07b00"
DONE_COLOR = "#3a6fb0"
PAD_FACE = "#dddddd"
IMG_FACE = "#ffffff"

STEPS = [(i, j) for i in range(KSIZE) for j in range(KSIZE)]  # row-major, matches the code's nested loop


def fmt(v):
    """Format a number, avoiding the ugly "-0" that :g produces for -0.0."""
    if abs(v) < 1e-9:
        return "0"
    return f"{v:g}"


def draw_small_grid(ax, M, title, border_color, active_ij=None, cell_text_color="#222222"):
    k = M.shape[0]
    ax.imshow(np.ones((k, k, 3)), interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which="minor", color="#888888", linewidth=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for r in range(k):
        for c in range(k):
            color = ACTIVE_COLOR if active_ij == (r, c) else cell_text_color
            weight = "bold" if active_ij == (r, c) else "normal"
            ax.text(c, r, fmt(M[r, c]), ha="center", va="center",
                     fontsize=10, color=color, fontweight=weight)
    if active_ij is not None:
        r, c = active_ij
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                 edgecolor=ACTIVE_COLOR, linewidth=2.6, zorder=3))
    ax.set_xlim(-0.5, k - 0.5)
    ax.set_ylim(k - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    ax.set_title(title, fontsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(2.2)


def draw_padded_panel(ax, window_ij=None):
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
            ax.text(cc, rr, fmt(padded[rr, cc]), ha="center", va="center",
                     fontsize=8, color="#222222", zorder=1)

    if window_ij is not None:
        i, j = window_ij
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), W, H, fill=False,
                                 edgecolor=ACTIVE_COLOR, linewidth=2.6, zorder=3))

    ax.set_title("padded image (shifted window outlined)", fontsize=11)


def draw_value_panel(ax, M, title, cell_face="#ffffff", text_color="#222222"):
    for r in range(H):
        for c in range(W):
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=cell_face,
                                     edgecolor="#aaaaaa", linewidth=0.7, zorder=0))
            ax.text(c, r, fmt(M[r, c]), ha="center", va="center", fontsize=8.5,
                     color=text_color, fontweight="bold", zorder=1)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.tick_params(length=0, labelsize=7)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    ax.set_title(title, fontsize=11)


def render(step, final_hold=False):
    """step: -1 for the blank intro, 0..8 for the kernel entry currently
    being applied, or 9 once all nine have been added in."""
    total = len(STEPS)
    is_stepping = 0 <= step < total

    fig, (ax_kernel, ax_padded, ax_contrib, ax_accum) = plt.subplots(
        1, 4, figsize=(16.5, 5.0), dpi=120,
        gridspec_kw={"width_ratios": [0.42, 0.85, 0.72, 0.72]},
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.78, bottom=0.07, wspace=0.32)

    active_ij = STEPS[step] if is_stepping else None
    draw_small_grid(ax_kernel, flipped, "flipped kernel", ACTIVE_COLOR, active_ij=active_ij,
                     cell_text_color="#555555")

    draw_padded_panel(ax_padded, window_ij=active_ij)

    accumulator = np.zeros((H, W))
    upto = max(step, 0) if step >= 0 else 0
    for k in range(min(upto, total)):
        i, j = STEPS[k]
        accumulator += flipped[i, j] * padded[i:i + H, j:j + W]

    if is_stepping:
        i, j = active_ij
        contribution = flipped[i, j] * padded[i:i + H, j:j + W]
        draw_value_panel(ax_contrib, contribution, f"weight × window = {flipped[i, j]:g} × image slice",
                          cell_face="#fff1e0", text_color=ACTIVE_COLOR)
    else:
        draw_value_panel(ax_contrib, np.zeros((H, W)), "weight × window", cell_face="#ffffff", text_color="#cccccc")

    accum_face = "#eef3fa" if upto > 0 else "#ffffff"
    accum_color = DONE_COLOR if upto > 0 else "#cccccc"
    draw_value_panel(ax_accum, accumulator, "accumulator (running sum)", cell_face=accum_face, text_color=accum_color)

    if is_stepping:
        i, j = active_ij
        suptitle = f"step {step + 1}/{total}: kernel weight ({i},{j}) = {flipped[i, j]:g} → add {flipped[i, j]:g} × (shifted image) to the accumulator"
        color = ACTIVE_COLOR
    elif step < 0:
        suptitle = "loop over the 9 kernel weights (not the 25 image pixels)"
        color = "#222222"
    else:
        suptitle = ""
        color = "#222222"

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, color=color)

    if not is_stepping and step >= 0:
        ax_accum.text(0.5, 1.2, "FINAL RESULT", transform=ax_accum.transAxes,
                        ha="center", va="bottom", fontsize=15, fontweight="bold",
                        color="#1f1f1f")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# Sanity check: the accumulator trick must match the position-by-position definition.
final_accumulator = np.zeros((H, W))
for i, j in STEPS:
    final_accumulator += flipped[i, j] * padded[i:i + H, j:j + W]
assert np.allclose(final_accumulator, reference_output), "accumulator trick does not match direct definition!"

frames = []
HOLD_INTRO = 10
HOLD_STEP = 16
HOLD_END = 28

frames += [render(-1)] * HOLD_INTRO

for step in range(len(STEPS)):
    frames += [render(step)] * HOLD_STEP

frames += [render(len(STEPS), final_hold=True)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=110,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames, {len(STEPS)} kernel weights)")
