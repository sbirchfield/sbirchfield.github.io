"""Generate roi_align_demo.gif: a conceptual companion to the lesson 43
notebook's "Mask R-CNN: Faster R-CNN plus a mask branch" section, illustrating
the RoIPool-vs-RoIAlign distinction described there: RoIPool quantizes a
proposed box's coordinates to the nearest feature-map cell before extracting
features, while RoIAlign reads features via bilinear interpolation at the
box's true (non-quantized) coordinates.

Standalone (not tied to either lesson's specific dataset) -- a small 8x8
coarse feature grid with a box sliding through sub-pixel offsets, same
sliding-box pattern as lesson 41's iou_demo.

Run from anywhere:
    python make_roi_align_demo.py
Output:
    roi_align_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "roi_align_demo.gif")

TEXT_COLOR = "#222222"
BOX_COLOR = "#1f6fb4"
ROIPOOL_COLOR = "#f2994a"   # rounded/quantized -- imprecise
ROIALIGN_COLOR = "#2e9e44"  # bilinear-sampled -- stays aligned

GRID = 8
BOX_W = BOX_H = 2.6
Y0 = 2.5  # fixed vertical position; only x slides

# A smooth gradient plus a few bumps, so bilinear-interpolated sample values
# visibly differ from nearest-neighbor quantized ones.
gy, gx = np.mgrid[0:GRID, 0:GRID]
feature_grid = (gx * 0.9 + gy * 0.5 + 2.0 * np.sin(gx * 0.8) * np.cos(gy * 0.6)).astype(np.float32)

OFFSETS_X = list(np.arange(0.8, 4.61, 0.19))


def bilinear_sample(grid, x, y):
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, GRID - 1), min(y0 + 1, GRID - 1)
    x0, y0 = max(x0, 0), max(y0, 0)
    fx, fy = x - x0, y - y0
    v00, v10 = grid[y0, x0], grid[y0, x1]
    v01, v11 = grid[y1, x0], grid[y1, x1]
    return (v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy)
            + v01 * (1 - fx) * fy + v11 * fx * fy)


SAMPLE_PTS = [(0.2, 0.2), (0.5, 0.2), (0.8, 0.2),
              (0.2, 0.5), (0.5, 0.5), (0.8, 0.5),
              (0.2, 0.8), (0.5, 0.8), (0.8, 0.8)]  # 3x3 grid, fractional position within the box


def panel(ax, x0, show_align):
    ax.imshow(feature_grid, cmap="Blues", vmin=feature_grid.min(), vmax=feature_grid.max(), origin="upper")
    ax.set_xticks(np.arange(-0.5, GRID, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.set_xticks([]); ax.set_yticks([])

    # True (sub-pixel) box, in image/array coordinates: imshow's x/y match array col/row.
    ax.add_patch(patches.Rectangle((x0 - 0.5, Y0 - 0.5), BOX_W, BOX_H, edgecolor=BOX_COLOR,
                                    facecolor="none", linewidth=2.6, zorder=5))

    if show_align:
        for fx, fy in SAMPLE_PTS:
            sx = x0 + fx * BOX_W
            sy = Y0 + fy * BOX_H
            ax.plot(sx - 0.5, sy - 0.5, marker="o", color=ROIALIGN_COLOR, markersize=6,
                     markeredgecolor="white", markeredgewidth=0.8, zorder=6)
    else:
        qx0 = round(x0); qy0 = round(Y0)
        qx1 = round(x0 + BOX_W); qy1 = round(Y0 + BOX_H)
        ax.add_patch(patches.Rectangle((qx0 - 0.5, qy0 - 0.5), qx1 - qx0, qy1 - qy0, edgecolor=ROIPOOL_COLOR,
                                        facecolor=ROIPOOL_COLOR, alpha=0.35, linewidth=2.2,
                                        linestyle="--", zorder=4))


def render(x0):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6), dpi=120)
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.03, right=0.97, wspace=0.12)

    panel(axes[0], x0, show_align=False)
    axes[0].set_title("RoIPool: coordinates rounded\nto the nearest feature cell", fontsize=9.5, color=TEXT_COLOR)

    panel(axes[1], x0, show_align=True)
    axes[1].set_title("RoIAlign: features bilinearly\nsampled at the true coordinates", fontsize=9.5,
                       color=TEXT_COLOR)

    fig.suptitle("RoIPool vs. RoIAlign", fontsize=13, fontweight="bold", color=TEXT_COLOR, y=0.97)
    fig.text(0.5, 0.06, f"box left edge = {x0:.2f}  (feature-grid units)", ha="center", fontsize=9,
              color=TEXT_COLOR)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
for x0 in OFFSETS_X:
    frames += [render(x0)] * 9
frames += [render(OFFSETS_X[-1])] * 60

frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=55, loop=0)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
