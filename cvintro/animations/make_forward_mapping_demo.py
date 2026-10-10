"""Generate forward_mapping_demo.gif: an animated version of the lesson 9
notebook's "The problem with forward mapping" cell. Instead of only showing
the final result (holes already present), this scans the source image row
by row and splats each source pixel into the destination canvas, so the
viewer watches the gaps open up in real time as the enlarging/rotating
transform spreads the destination locations apart.

Uses the same test image and transform as the notebook cell.

Run from anywhere:
    python make_forward_mapping_demo.py
Output:
    forward_mapping_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "forward_mapping_demo.gif")

ACTIVE_COLOR = "#e07b00"
WARN_COLOR = "#c21807"


def make_test_image(size=60):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.rectangle(img, (5, 5), (size - 5, size - 5), (60, 90, 160), -1)
    tri = np.array([[30, 12], [10, 32], [50, 32]], dtype=np.int32)
    cv2.fillPoly(img, [tri], (225, 195, 60))
    return img


img = make_test_image()
h, w = img.shape[:2]
M = cv2.getRotationMatrix2D((w / 2, h / 2), angle=25, scale=1.6)

ys, xs = np.mgrid[0:h, 0:w]
src_pts = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
dst_pts = M @ src_pts
dxi_all = np.round(dst_pts[0]).astype(int).reshape(h, w)
dyi_all = np.round(dst_pts[1]).astype(int).reshape(h, w)


def render(n_rows, final_hold=False):
    forward = np.zeros_like(img)
    filled_mask = np.zeros((h, w), dtype=bool)
    if n_rows > 0:
        dxi = dxi_all[:n_rows].ravel()
        dyi = dyi_all[:n_rows].ravel()
        src_x = xs[:n_rows].ravel()
        src_y = ys[:n_rows].ravel()
        valid = (dxi >= 0) & (dxi < w) & (dyi >= 0) & (dyi < h)
        forward[dyi[valid], dxi[valid]] = img[src_y[valid], src_x[valid]]
        filled_mask[dyi[valid], dxi[valid]] = True

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.2), dpi=120)
    fig.subplots_adjust(top=0.80, bottom=0.14, wspace=0.15)
    ax_src, ax_dst = axes

    ax_src.imshow(img, interpolation="nearest")
    if n_rows < h and not final_hold:
        ax_src.axhspan(n_rows - 0.5, n_rows + 0.5, color=ACTIVE_COLOR, alpha=0.6)
    ax_src.set_title(f"source: row {min(n_rows, h)} / {h}", fontsize=10.5)
    ax_src.set_xticks([])
    ax_src.set_yticks([])

    ax_dst.imshow(forward, interpolation="nearest")
    ax_dst.set_title("destination (forward mapping)", fontsize=10.5)
    ax_dst.set_xticks([])
    ax_dst.set_yticks([])

    total = h * w
    filled = int(filled_mask.sum())
    if final_hold:
        suptitle = f"{filled} / {total} destination pixels filled ({100 * filled / total:.0f}%) -- the rest are holes"
        color = WARN_COLOR
    else:
        suptitle = "forward mapping: push each source pixel to its transformed location"
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
