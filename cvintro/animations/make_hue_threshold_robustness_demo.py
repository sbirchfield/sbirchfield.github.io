"""Generate hue_threshold_robustness_demo.gif: an animated version of the
lesson 18 notebook's HSV-vs-RGB segmentation cell. Reuses that cell's exact
disk image and threshold values, but animates the lighting gradient's
strength from flat (uniform lighting) to the notebook's own steep
left-dark/right-bright gradient, with the HSV hue mask and fixed RGB-range
mask recomputed live each frame -- the HSV mask stays locked onto the disk
throughout, while the RGB mask visibly erodes from the dark side as the
gradient steepens.

Run from anywhere:
    python make_hue_threshold_robustness_demo.py
Output:
    hue_threshold_robustness_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "hue_threshold_robustness_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
FAIL_COLOR = "#c0392b"

size = 200
flat_img = np.zeros((size, size, 3), dtype=np.uint8)
cv2.circle(flat_img, (100, 100), 70, (0, 100, 255), -1)  # an orange disk, BGR

yy, xx = np.mgrid[0:size, 0:size]
true_mask = (xx - 100) ** 2 + (yy - 100) ** 2 <= 70 ** 2


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


def shaded(t):
    """t=0: uniform lighting. t=1: the notebook's own gradient
    (0.25 + 0.9*(xx/size)), dark on the left, bright on the right."""
    lo = 1.0 - 0.75 * t
    hi = 1.0 + 0.15 * t
    gradient = lo + (hi - lo) * (xx / size)
    return np.clip(flat_img.astype(np.float64) * gradient[..., None], 0, 255).astype(np.uint8)


def render(t):
    shaded_disk = shaded(t)
    hsv = cv2.cvtColor(shaded_disk, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]

    mask_hsv = (hue > 5) & (hue < 25)
    mask_rgb = ((shaded_disk[:, :, 2] > 150) & (shaded_disk[:, :, 1] > 50) &
                (shaded_disk[:, :, 1] < 180) & (shaded_disk[:, :, 0] < 80))

    iou_rgb = iou(mask_rgb, true_mask)

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.5), dpi=120)
    axes[0].imshow(cv2.cvtColor(shaded_disk, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"shaded disk (gradient strength {t:.2f})", fontsize=9.5, color=ACTIVE_COLOR)

    axes[1].imshow(mask_hsv, cmap="gray")
    axes[1].set_title("HSV hue threshold", fontsize=9.5, color=MATCH_COLOR, fontweight="bold")

    axes[2].imshow(mask_rgb, cmap="gray")
    rgb_color = MATCH_COLOR if iou_rgb > 0.85 else FAIL_COLOR
    axes[2].set_title("fixed RGB range", fontsize=9.5, color=rgb_color, fontweight="bold")

    for ax in axes:
        ax.axis("off")
    fig.suptitle("HSV hue threshold stays locked onto the disk despite shading",
                 fontsize=10.5, color="#222222")
    fig.subplots_adjust(top=0.86, bottom=0.02, wspace=0.12)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


N = 40
ts = np.linspace(0, 1, N)
frames = [render(t) for t in ts]
frames += [frames[-1]] * 16

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=90,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
