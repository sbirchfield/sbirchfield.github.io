"""Generate bayer_demosaic_demo.gif: an animated version of the lesson 26
notebook's Bayer mosaic + bilinear demosaicking cells. Zooms into the same
16x16 RGGB mosaic crop used there, then fills in each pixel's two missing
colors one pixel at a time (raster order), using the notebook's own
demosaick_bilinear_rggb() result for the final value -- so the viewer watches
the sparse, single-channel-per-pixel mosaic turn into a full-color image
pixel by pixel, instead of seeing only the before/after.

Run from anywhere:
    python make_bayer_demosaic_demo.py
Output:
    bayer_demosaic_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "bayer_demosaic_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"

# Same synthetic scene, mosaic, and crop as the notebook cells.
rng = np.random.default_rng(0)
bayer_scene = np.zeros((120, 160, 3), dtype=np.uint8)
cv2.rectangle(bayer_scene, (10, 10), (70, 100), (40, 180, 220), -1)
cv2.circle(bayer_scene, (110, 60), 40, (200, 90, 40), -1)
bayer_scene = np.clip(bayer_scene.astype(np.float64) + rng.normal(0, 4, bayer_scene.shape), 0, 255).astype(np.uint8)


def make_bayer_rggb(img):
    H, W = img.shape[:2]
    mosaic = np.zeros((H, W), dtype=np.uint8)
    mosaic[0::2, 0::2] = img[0::2, 0::2, 2]
    mosaic[0::2, 1::2] = img[0::2, 1::2, 1]
    mosaic[1::2, 0::2] = img[1::2, 0::2, 1]
    mosaic[1::2, 1::2] = img[1::2, 1::2, 0]
    return mosaic


def colorize_bayer_rggb(mosaic):
    H, W = mosaic.shape
    colored = np.zeros((H, W, 3), dtype=np.uint8)
    colored[0::2, 0::2, 0] = mosaic[0::2, 0::2]
    colored[0::2, 1::2, 1] = mosaic[0::2, 1::2]
    colored[1::2, 0::2, 1] = mosaic[1::2, 0::2]
    colored[1::2, 1::2, 2] = mosaic[1::2, 1::2]
    return colored


def demosaic_bilinear_rggb(mosaic):
    H, W = mosaic.shape
    R = np.zeros((H, W)); G = np.zeros((H, W)); B = np.zeros((H, W))
    R[0::2, 0::2] = mosaic[0::2, 0::2]
    G[0::2, 1::2] = mosaic[0::2, 1::2]; G[1::2, 0::2] = mosaic[1::2, 0::2]
    B[1::2, 1::2] = mosaic[1::2, 1::2]

    Rmask = np.zeros((H, W)); Rmask[0::2, 0::2] = 1
    Gmask = np.zeros((H, W)); Gmask[0::2, 1::2] = 1; Gmask[1::2, 0::2] = 1
    Bmask = np.zeros((H, W)); Bmask[1::2, 1::2] = 1

    def fill(channel, mask):
        kernel = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])
        weight_sum = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_REFLECT)
        value_sum = cv2.filter2D(channel * mask, -1, kernel, borderType=cv2.BORDER_REFLECT)
        filled = channel.copy()
        empty = mask == 0
        filled[empty] = value_sum[empty] / np.clip(weight_sum[empty], 1e-6, None)
        return filled

    R_full, G_full, B_full = fill(R, Rmask), fill(G, Gmask), fill(B, Bmask)
    return np.clip(np.stack([B_full, G_full, R_full], axis=-1), 0, 255).astype(np.uint8)


bayer = make_bayer_rggb(bayer_scene)
bayer_colored = colorize_bayer_rggb(bayer)
# demosaic_bilinear_rggb returns BGR order (matching the notebook's own
# convention, which converts before imshow) -- convert once here too, or the
# reconstructed colors come out with red and blue swapped.
demosaicked = cv2.cvtColor(demosaic_bilinear_rggb(bayer), cv2.COLOR_BGR2RGB)

# Straddles the boundary between the gold rectangle and the blue circle
# (with a sliver of background), instead of sitting inside one flat color.
Y0, Y1, X0, X1 = 52, 68, 62, 78
CROP_RAW = bayer_colored[Y0:Y1, X0:X1]
CROP_FULL = demosaicked[Y0:Y1, X0:X1]
N = Y1 - Y0  # 16x16

ZOOM = 24


def zoomed(img):
    return cv2.resize(img, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)


def render(filled_count, highlight_rc=None):
    """filled_count: how many pixels (raster order) already show their final
    demosaicked color; the rest still show the raw sparse (single-channel)
    reading."""
    current = CROP_RAW.copy()
    flat_idx = 0
    for r in range(N):
        for c in range(N):
            if flat_idx < filled_count:
                current[r, c] = CROP_FULL[r, c]
            flat_idx += 1

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.8), dpi=120)
    fig.subplots_adjust(top=0.80, bottom=0.04, wspace=0.12)

    axes[0].imshow(zoomed(CROP_RAW))
    axes[0].set_title("raw mosaic\n(1 known color per pixel)", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(zoomed(current))
    n_total = N * N
    color = MATCH_COLOR if filled_count >= n_total else ACTIVE_COLOR
    axes[1].set_title(f"filling in missing colors\n{filled_count:3d}/{n_total} pixels done", fontsize=10, color=color)
    axes[1].axis("off")
    if highlight_rc is not None:
        r, c = highlight_rc
        axes[1].add_patch(plt.Rectangle((c * ZOOM, r * ZOOM), ZOOM, ZOOM, fill=False,
                                          edgecolor=ACTIVE_COLOR, linewidth=2.5))

    axes[2].imshow(zoomed(CROP_FULL))
    axes[2].set_title("fully demosaicked", fontsize=10)
    axes[2].axis("off")

    fig.suptitle("bilinear demosaicking: each pixel keeps its own color, blends in the other two",
                  fontsize=12, color=NEUTRAL_COLOR, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_START = 14
STEP_HOLD = 3
HOLD_END = 44
BATCH = 4  # fill this many pixels (raster order) per animation step

frames += [render(0)] * HOLD_START

total = N * N
for i in range(0, total, BATCH):
    done = min(i + BATCH, total)
    r, c = divmod(done - 1, N)
    frames += [render(done, highlight_rc=(r, c))] * STEP_HOLD

frames += [render(total)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=45,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
