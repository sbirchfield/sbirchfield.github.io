"""Generate block_matching_demo.gif: an animated version of the lesson 22
notebook's "Block matching along the scanline" cell. For one fixed pixel in
the left crop, slides the search window along the same row of the right
crop over increasing candidate disparities, plotting the sum-of-squared-
differences cost curve live and marking the running best-so-far disparity,
until the true minimum (matching the ground-truth disparity) is found.

Reuses the exact synthetic stereo pair and block-matching formula from the
notebook (same rng seed, same disparity planes, same crop, same block_size
and max_disp) so the animation is visually consistent with the static cells
it accompanies.

Run from anywhere:
    python make_block_matching_demo.py
Output:
    block_matching_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

def box_corner(xc, yc):
    half = BLOCK_SIZE / 2
    return (xc - half - 0.5, yc - half - 0.5), BLOCK_SIZE


def add_halo(ax, xc, yc, linewidth):
    """White background rectangle, drawn before any colored box so that
    colored outlines always end up on top of every halo, never the reverse."""
    corner, size = box_corner(xc, yc)
    ax.add_patch(Rectangle(corner, size, size, fill=False, edgecolor="white",
                            linewidth=linewidth + 2))


def add_outline(ax, xc, yc, edgecolor, linewidth, linestyle="-"):
    corner, size = box_corner(xc, yc)
    ax.add_patch(Rectangle(corner, size, size, fill=False, edgecolor=edgecolor,
                            linewidth=linewidth, linestyle=linestyle))

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "block_matching_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"

# Same synthetic stereo pair as the notebook's own cell (rng seed 0).
rng = np.random.default_rng(0)
h, w = 150, 200
left_full = rng.integers(0, 255, (h, w)).astype(np.uint8)

true_disparity = np.full((h, w), 5, dtype=np.int32)
true_disparity[30:120, 50:150] = 15
true_disparity[60:90, 80:120] = 25

right_full = np.full((h, w), -1, dtype=np.int32)
for y in range(h):
    for x in range(w):
        xr = x - true_disparity[y, x]
        if 0 <= xr < w:
            right_full[y, xr] = left_full[y, x]
occluded = right_full == -1
right_full[occluded] = rng.integers(0, 255, occluded.sum())
right_full = right_full.astype(np.uint8)

# Same crop as the notebook's block-matching cell.
CROP = np.s_[0:100, 0:100]
left = left_full[CROP]
right = right_full[CROP]

BLOCK_SIZE = 7
MAX_DISP = 30
HALF = BLOCK_SIZE // 2

# A pixel well inside the near (disparity=15) rectangle, with enough room on
# both sides to search the full disparity range without hitting the crop edge.
# (Picked for a clear cost-curve margin at the true disparity -- with only a
# 7x7 window of random texture, many interior pixels have a coincidentally
# lower-cost false match elsewhere, same as the notebook's own dense
# block_match_stereo() occasionally gets wrong in the interior by chance.)
PY, PX = 45, 70
TRUE_D = int(true_disparity[CROP][PY, PX])

left_f, right_f = left.astype(np.float64), right.astype(np.float64)
left_patch = left_f[PY - HALF:PY + HALF + 1, PX - HALF:PX + HALF + 1]

costs = []
for d in range(MAX_DISP + 1):
    xr = PX - d
    right_patch = right_f[PY - HALF:PY + HALF + 1, xr - HALF:xr + HALF + 1]
    costs.append(((left_patch - right_patch) ** 2).sum())
costs = np.array(costs)


def render(step, final_hold=False):
    d = min(step, MAX_DISP)
    xr = PX - d
    best_d = int(np.argmin(costs[:d + 1]))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.3), dpi=120,
                               gridspec_kw={"width_ratios": [1.0, 1.0, 1.1]})
    fig.subplots_adjust(top=0.82, bottom=0.14, wspace=0.3)
    ax_left, ax_right, ax_cost = axes

    ax_left.imshow(left, cmap="gray")
    add_halo(ax_left, PX, PY, 2.4)
    add_outline(ax_left, PX, PY, NEUTRAL_COLOR, 2.4)
    ax_left.set_title("left: fixed patch", fontsize=10)
    ax_left.set_xticks([]); ax_left.set_yticks([])

    ax_right.imshow(right, cmap="gray")
    best_color = MATCH_COLOR if (final_hold or d == MAX_DISP) else ACTIVE_COLOR
    cur_color = MATCH_COLOR if d == best_d else ACTIVE_COLOR
    xr_best = PX - best_d
    draw_cur = d != best_d or not final_hold
    # Both white halos first (background), then green (best), then orange (current) on top.
    if draw_cur:
        add_halo(ax_right, xr, PY, 2.4)
    add_halo(ax_right, xr_best, PY, 2.8)
    add_outline(ax_right, xr_best, PY, MATCH_COLOR, 2.8)
    if draw_cur:
        add_outline(ax_right, xr, PY, cur_color, 2.4)
    ax_right.set_title(f"right: sliding window  (d={d:2d})", fontsize=10, color=cur_color)
    ax_right.set_xticks([]); ax_right.set_yticks([])

    ax_cost.plot(range(d + 1), costs[:d + 1], color=NEUTRAL_COLOR, linewidth=1.2, zorder=2)
    ax_cost.scatter([d], [costs[d]], color=cur_color, s=60, zorder=3)
    ax_cost.scatter([best_d], [costs[best_d]], facecolors="none", edgecolors=MATCH_COLOR,
                     s=140, linewidths=1.8, zorder=4)
    ax_cost.axvline(TRUE_D, color="#cccccc", linewidth=1.0, linestyle=":", zorder=1)
    ax_cost.set_xlim(-1, MAX_DISP + 1)
    ax_cost.set_ylim(0, costs.max() * 1.08)
    ax_cost.set_yticks([])
    ax_cost.set_xlabel("candidate disparity d", fontsize=9)
    ax_cost.set_ylabel("SSD cost", fontsize=9)
    ax_cost.set_title(f"best so far: d={best_d:2d}", fontsize=10, color=MATCH_COLOR)

    if final_hold:
        title = f"found: d={best_d} (true disparity = {TRUE_D})"
        color = MATCH_COLOR
    else:
        title = f"searching candidate disparities 0..{MAX_DISP}"
        color = ACTIVE_COLOR
    fig.suptitle(title, fontsize=12.5, color=color, fontweight="bold" if final_hold else "normal")
    fig.text(0.5, 0.895, f"block matching: one pixel, {BLOCK_SIZE}x{BLOCK_SIZE} window, sliding along the scanline",
              ha="center", va="top", fontsize=9.5, color=NEUTRAL_COLOR)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 4
HOLD_END = 44

for step in range(MAX_DISP + 1):
    frames += [render(step)] * HOLD_STEP
frames += [render(MAX_DISP, final_hold=True)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=70,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames, true disparity={TRUE_D})")
