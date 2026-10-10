"""Generate inverse_mapping_demo.gif: the companion to forward_mapping_demo.gif.
Scans the *destination* image row by row (instead of the source), computing
each destination pixel's fractional source coordinate via the inverse
transform and nearest-neighbor sampling it, so the viewer watches the
destination fill in completely -- no holes, by construction -- in direct
contrast to forward mapping's speckled gaps.

Uses the same test image and transform as the notebook cells.

Run from anywhere:
    python make_inverse_mapping_demo.py
Output:
    inverse_mapping_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "inverse_mapping_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"


def make_test_image(size=60):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.rectangle(img, (5, 5), (size - 5, size - 5), (60, 90, 160), -1)
    tri = np.array([[30, 12], [10, 32], [50, 32]], dtype=np.int32)
    cv2.fillPoly(img, [tri], (225, 195, 60))
    return img


img = make_test_image()
h, w = img.shape[:2]
M = cv2.getRotationMatrix2D((w / 2, h / 2), angle=25, scale=1.6)
M_inv = cv2.invertAffineTransform(M)

ys, xs = np.mgrid[0:h, 0:w]
dst_grid = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
src_coords = M_inv @ dst_grid
src_x_all = src_coords[0].reshape(h, w)
src_y_all = src_coords[1].reshape(h, w)


def nearest_sample(image, xf, yf):
    hh, ww = image.shape[:2]
    xi = np.round(xf).astype(int)
    yi = np.round(yf).astype(int)
    valid = (xi >= 0) & (xi < ww) & (yi >= 0) & (yi < hh)
    out = np.zeros(xf.shape + (image.shape[2],), dtype=np.uint8)
    out[valid] = image[yi[valid], xi[valid]]
    return out


def render(n_rows, final_hold=False):
    dest = np.zeros_like(img)
    if n_rows > 0:
        dest[:n_rows] = nearest_sample(img, src_x_all[:n_rows], src_y_all[:n_rows])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.2), dpi=120)
    fig.subplots_adjust(top=0.80, bottom=0.14, wspace=0.15)
    ax_src, ax_dst = axes

    ax_src.imshow(img, interpolation="nearest")
    if 0 < n_rows <= h and not final_hold:
        row = n_rows - 1
        ax_src.scatter(src_x_all[row], src_y_all[row], s=4, color=ACTIVE_COLOR)
    ax_src.set_title("source (sampled at fractional coords)", fontsize=10.5)
    ax_src.set_xticks([])
    ax_src.set_yticks([])
    ax_src.set_xlim(-0.5, w - 0.5)
    ax_src.set_ylim(h - 0.5, -0.5)

    ax_dst.imshow(dest, interpolation="nearest")
    if n_rows < h and not final_hold:
        ax_dst.axhspan(n_rows - 0.5, n_rows + 0.5, color=ACTIVE_COLOR, alpha=0.6)
    ax_dst.set_title(f"destination: row {min(n_rows, h)} / {h}", fontsize=10.5)
    ax_dst.set_xticks([])
    ax_dst.set_yticks([])

    if final_hold:
        suptitle = f"{h * w} / {h * w} destination pixels filled (100%) -- no holes, by construction"
        color = MATCH_COLOR
    else:
        suptitle = "inverse mapping: for each destination pixel, sample from its source location"
        color = "#222222"
    fig.suptitle(suptitle, fontsize=12, color=color, fontweight="bold" if final_hold else "normal")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_INTRO = 10
HOLD_ROW = 3
HOLD_DONE = 14
HOLD_END = 44

ROWS_PER_STEP = 2

frames += [render(0)] * HOLD_INTRO
for n in range(ROWS_PER_STEP, h + 1, ROWS_PER_STEP):
    frames += [render(n)] * HOLD_ROW
frames += [render(h)] * HOLD_DONE
frames += [render(h, final_hold=True)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=90,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
