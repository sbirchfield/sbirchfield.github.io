"""Generate sift_scalespace_demo.gif: an animated companion to the lesson 20
notebook's "SIFT: scale-space extrema" cell. Builds a Difference-of-Gaussians
(DoG) scale-space (geometrically increasing sigma) on a synthetic image with
two blobs of different radii, then shows the actual 26-neighbor extremum test
SIFT keypoint detection relies on: a candidate pixel slides across the image
(left panel) while, on the right, a box tracks wherever the smallest of its
27 values (itself + 8 same-scale neighbors + 9+9 cross-scale neighbors)
currently is. The box is orange while that minimum is some neighbor, and
lights up green exactly when it lands on the candidate pixel itself -- a
confirmed local extremum.

Both blobs are genuine scale-space extrema, but at *different* scales -- the
small blob's response peaks (most negative, since it's brighter than the
background) at a smaller sigma than the large blob's, directly illustrating
why DoG scale-space search finds the "right" scale per feature rather than
needing one fixed window size (the problem the previous cell raised).

Run from anywhere:
    python make_sift_scalespace_demo.py
Output:
    sift_scalespace_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "sift_scalespace_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"

SIZE = 200
img = np.full((SIZE, SIZE), 50, dtype=np.float64)
yy, xx = np.mgrid[0:SIZE, 0:SIZE]


def add_blob(cx, cy, r, val):
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return val * np.exp(-d2 / (2 * r ** 2))


BLOB1 = dict(cx=60, cy=100, r=8)   # small blob
BLOB2 = dict(cx=140, cy=100, r=16)  # large blob, radius 2x
img += add_blob(BLOB1["cx"], BLOB1["cy"], BLOB1["r"], 180)
img += add_blob(BLOB2["cx"], BLOB2["cy"], BLOB2["r"], 180)
img = np.clip(img, 0, 255)

N_LEVELS = 9
SIGMAS = [1.0 * 1.5 ** i for i in range(N_LEVELS)]
gaussians = [cv2.GaussianBlur(img, (0, 0), s) for s in SIGMAS]
# N_LEVELS Gaussians -> N_LEVELS-1 DoG levels, same convention as the notebook.
dog = [gaussians[i + 1] - gaussians[i] for i in range(N_LEVELS - 1)]

DOG_VMIN, DOG_VMAX = -40, 10  # shared normalization across all levels/patches


def patch(level, cx, cy):
    return dog[level][cy - 1:cy + 2, cx - 1:cx + 2]


def is_scalespace_min(level, cx, cy):
    """True iff dog[level][cy,cx] is strictly less than all 26 neighbors
    (8 same-scale + 9 below + 9 above) -- the actual SIFT keypoint test."""
    center = dog[level][cy, cx]
    neighborhood = np.concatenate([
        patch(level - 1, cx, cy).ravel(),
        patch(level, cx, cy).ravel(),
        patch(level + 1, cx, cy).ravel(),
    ])
    others = np.delete(neighborhood, 13)  # drop the center pixel itself
    return center < others.min()


# Only levels 3-7 are ever used by the two sweep legs below (level +/- 1 of
# 5 and of 6) -- the bottom 3, smallest-sigma levels are never compared
# against, so they're dropped from the displayed stack entirely.
DISPLAY_LEVELS = list(range(3, len(dog)))


def draw_stack(ax, cx, cy, level):
    """Vertical strip of each displayed DoG level's thumbnail, with the
    current level (and its two scale-neighbors) highlighted and the
    candidate pixel marked. The dot moves with cx as the candidate slides
    across the image."""
    ax.set_xlim(0, 1)
    ax.set_ylim(len(DISPLAY_LEVELS), 0)
    ax.axis("off")
    thumb_h = 0.92
    px = 0.15 + 0.7 * (cx / SIZE)
    for row, i in enumerate(DISPLAY_LEVELS):
        d = dog[i]
        ax.imshow(d, cmap="gray", vmin=DOG_VMIN, vmax=DOG_VMAX,
                   extent=[0.15, 0.85, row + thumb_h, row + (1 - thumb_h)], zorder=1)
        ax.text(0.03, row + 0.5, f"$\\sigma=${SIGMAS[i]:.1f}", fontsize=7.5, ha="right", va="center",
                 color=NEUTRAL_COLOR)
        if i == level:
            color, lw = ACTIVE_COLOR, 2.4
        elif abs(i - level) == 1:
            color, lw = "#bbbbbb", 1.6
        else:
            color, lw = "none", 0
        if color != "none":
            ax.add_patch(Rectangle((0.15, row + (1 - thumb_h)), 0.7, thumb_h - (1 - thumb_h),
                                     fill=False, edgecolor=color, linewidth=lw))
            # mark the candidate's (x, y) location only on the 3 levels
            # actually being compared on the right, in the same color as
            # that level's border -- ties the two panels together instead
            # of scattering an unexplained dot on every thumbnail
            py = row + (1 - thumb_h) + thumb_h * (cy / SIZE)
            ax.scatter([px], [py], s=16, color=color, zorder=2)
    ax.set_title("DoG scale-space\ndot = candidate (x, y), sliding across the image", fontsize=9.5)


def argmin_location(cx, cy, level):
    """Where, among all 27 values (the center pixel + its 26 neighbors), the
    smallest one actually is -- (patch_index, row, col), patch_index 0/1/2
    meaning scale below/candidate level/scale above."""
    stacked = np.stack([patch(level - 1, cx, cy), patch(level, cx, cy), patch(level + 1, cx, cy)])
    return np.unravel_index(np.argmin(stacked), stacked.shape)


def draw_patches(fig, gs_row, cx, cy, level):
    """Draws the 3 same-scale-neighborhood patches with their raw values, and
    a single box tracking wherever the smallest of the 27 values currently
    is -- orange while that's some neighbor, green exactly when it lands on
    the candidate pixel itself (a confirmed local extremum)."""
    labels = ["scale below", "candidate level", "scale above"]
    min_patch, min_r, min_c = argmin_location(cx, cy, level)
    is_extremum = (min_patch, min_r, min_c) == (1, 1, 1)
    box_color = MATCH_COLOR if is_extremum else ACTIVE_COLOR
    for col, lvl in enumerate([level - 1, level, level + 1]):
        ax = fig.add_subplot(gs_row[col])
        p = patch(lvl, cx, cy)
        ax.imshow(p, cmap="coolwarm_r", vmin=DOG_VMIN, vmax=DOG_VMAX)
        is_center_col = lvl == level
        for r in range(3):
            for c in range(3):
                is_min_cell = col == min_patch and r == min_r and c == min_c
                # one decimal place: at this blob radius the 26 values differ
                # by only a fraction of a unit, so rounding to whole numbers
                # made every cell look identical even though the center is a
                # genuine (if narrow) minimum -- this makes that margin visible
                ax.text(c, r, f"{p[r, c]:.1f}", ha="center", va="center",
                         fontsize=7.5 if not is_min_cell else 8.5,
                         fontweight="bold" if is_min_cell else "normal",
                         color=box_color if is_min_cell else "black")
                if is_min_cell:
                    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                             edgecolor=box_color, linewidth=2.6, zorder=3))
        thin_border = ACTIVE_COLOR if is_center_col else "none"
        if thin_border != "none":
            ax.add_patch(Rectangle((-0.5, -0.5), 3, 3, fill=False, edgecolor=thin_border, linewidth=1.0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{labels[col]}\n($\\sigma=${SIGMAS[lvl]:.1f})", fontsize=8.5)
    return is_extremum


def render(cx, cy, level):
    fig = plt.figure(figsize=(9.6, 5.4), dpi=120)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.5, 1.0], left=0.1, right=0.97,
                           top=0.82, bottom=0.06, wspace=0.3)
    ax_stack = fig.add_subplot(gs[0])
    draw_stack(ax_stack, cx, cy, level)

    gs_patches = gs[1].subgridspec(1, 3, wspace=0.15)
    is_extremum = draw_patches(fig, gs_patches, cx, cy, level)

    if is_extremum:
        verdict = "box has landed on the center -> local extremum, SIFT keypoint"
        color = MATCH_COLOR
    else:
        verdict = "orange box = smallest of the 27 values; searching for it to land on the center"
        color = ACTIVE_COLOR
    fig.suptitle(verdict, fontsize=11.5, color=color, fontweight="bold" if is_extremum else "normal")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# Sweep the candidate pixel horizontally through each blob at its own matching
# scale -- verified offline that the minimum of the 27 values only lands
# exactly on the center pixel (a true local extremum) right at each blob's
# own center, and only at its own best-matching scale: the small blob
# (radius 8) at level 5 (sigma=7.59), the large blob (radius 16, 2x the
# radius) at level 6 (sigma=11.39) -- one octave later, tracking its size.
# Everywhere else along the sweep the box lands on some neighbor instead,
# which is the normal case -- most candidate pixels are not extrema.
LEGS = [
    (30, 90, 100, 5, BLOB1["cx"]),
    (110, 170, 100, 6, BLOB2["cx"]),
]
for x0, x1, cy, level, peak_cx in LEGS:
    assert is_scalespace_min(level, peak_cx, cy), f"expected extremum at x={peak_cx}, level={level}"

STEPS_PER_LEG = 36
HOLD_END = 14
HOLD_PEAK = 20

frames = []
for x0, x1, cy, level, peak_cx in LEGS:
    # Build the sweep from two linspaces meeting exactly at peak_cx (a plain
    # linspace(x0, x1, N) rounded to int generally never lands on peak_cx,
    # so the box would approach the center but never actually snap to it).
    xs = np.concatenate([
        np.linspace(x0, peak_cx, STEPS_PER_LEG // 2, endpoint=False),
        np.linspace(peak_cx, x1, STEPS_PER_LEG // 2),
    ])
    frames += [render(int(round(xs[0])), cy, level)] * HOLD_END
    for x in xs:
        cx = int(round(x))
        frames.append(render(cx, cy, level))
        if cx == peak_cx:
            frames += [render(cx, cy, level)] * HOLD_PEAK
    frames += [render(int(round(xs[-1])), cy, level)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=90,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
