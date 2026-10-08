"""Generate erosion_demo.gif: an animated, step-by-step walkthrough of binary
erosion with a 3x3 kernel on a small binary image, by hand.

Run from anywhere:
    python make_erosion_demo.py
Output:
    erosion_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "erosion_demo.gif")

# A small binary image with three features that erosion treats differently:
# a solid blob (shrinks but survives), a thin one-pixel-wide protrusion
# (removed entirely), and an isolated speck (removed entirely).
im = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.uint8)

H, W = im.shape
ys, xs = np.nonzero(im)  # erosion can only ever keep a foreground pixel, so
order = np.lexsort((xs, ys))  # we only need to visit foreground pixels, in
ys, xs = ys[order], xs[order]  # reading order, same shortcut as the moments demo.

PASS_COLOR = "#2e9e44"   # kernel outline / verdict: all 9 neighbors white
FAIL_COLOR = "#e54040"   # kernel outline / verdict: at least one neighbor is black
KERNEL_LW = 2.6


def survives(r, c):
    """True if every pixel in the 3x3 neighborhood of (r, c) is foreground,
    treating anything outside the image as background (zero-padding)."""
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= H or cc < 0 or cc >= W or im[rr, cc] == 0:
                return False
    return True


def draw_grid(ax, title):
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="#888888", linewidth=0.8)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.tick_params(length=0)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    ax.set_title(title, fontsize=13)


def draw_kernel_panel(ax):
    """Static panel shown every frame: the 3x3 structuring element itself,
    spelled out explicitly as a grid of 1s so there is no ambiguity about
    its shape (e.g. a full square, not a cross)."""
    ax.imshow(np.ones((3, 3, 3)), interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="#888888", linewidth=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for r in range(3):
        for c in range(3):
            ax.text(c, r, "1", ha="center", va="center", fontsize=12)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    ax.set_title("structuring\nelement", fontsize=11)


def render(step, final_hold=False):
    """step: -1 for the blank intro (no kernel shown yet), an index into
    (xs, ys) for the foreground pixel currently being tested, or len(xs)
    once every foreground pixel has been tested (final result)."""
    is_stepping = 0 <= step < len(xs)
    included = min(step + 1, len(xs)) if step >= 0 else 0

    out = np.zeros((H, W), dtype=np.uint8)
    for i in range(included):
        if survives(ys[i], xs[i]):
            out[ys[i], xs[i]] = 1

    in_rgb = np.zeros((H, W, 3))
    in_rgb[im == 1] = 1.0
    out_rgb = np.zeros((H, W, 3))
    out_rgb[out == 1] = 1.0

    fig, (ax_kernel, ax_in, ax_out) = plt.subplots(
        1, 3, figsize=(10.6, 4.6), dpi=120,
        gridspec_kw={"width_ratios": [0.3, 1.0, 1.0]},
    )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.8, bottom=0.08, wspace=0.22)

    draw_kernel_panel(ax_kernel)

    ax_in.imshow(in_rgb, interpolation="nearest", vmin=0, vmax=1)
    draw_grid(ax_in, "input")
    ax_out.imshow(out_rgb, interpolation="nearest", vmin=0, vmax=1)
    draw_grid(ax_out, "output (eroded)")

    if is_stepping:
        r, c = ys[step], xs[step]
        ok = survives(r, c)
        color = PASS_COLOR if ok else FAIL_COLOR
        ax_in.add_patch(Rectangle((c - 1.5, r - 1.5), 3, 3, fill=False,
                                   edgecolor=color, linewidth=KERNEL_LW, zorder=5))
        ax_in.add_patch(plt.Circle((c, r), 0.14, facecolor=color,
                                    edgecolor="none", zorder=6))
        verdict = "all 9 neighbors white -> KEEP" if ok else "some neighbor is black -> REMOVE"
        suptitle = f"checking pixel (x={c}, y={r}): {verdict}"
    elif step < 0:
        suptitle = "3x3 kernel will slide over every white pixel"
        color = "black"
    else:
        suptitle = ""
        color = "black"

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, color=color if is_stepping else "#222222")

    if not is_stepping and step >= 0:
        ax_out.text(0.5, 1.22, "FINAL RESULT", transform=ax_out.transAxes,
                     ha="center", va="bottom", fontsize=15, fontweight="bold",
                     color="#1f1f1f")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_INTRO = 10
HOLD_STEP = 12
HOLD_END = 24

intro = render(-1)
frames += [intro] * HOLD_INTRO

for step in range(len(xs)):
    frames += [render(step)] * HOLD_STEP

final = render(len(xs))
frames += [final] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=90,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames, {len(xs)} foreground pixels)")
