"""Generate lowpass_cutoff_demo.gif: an animated version of the lesson 15
notebook's ideal low-pass filtering cells. Sweeps the ideal low-pass cutoff
radius from large to small and back to large (a continuous loop) on the
notebook's own `photo` test image (a square plus a circle), with the
filtered image updating live -- it gets progressively blurrier, and ringing
artifacts near the sharp edges become visible as the cutoff tightens, then
fade again as the radius grows back.

Run from anywhere:
    python make_lowpass_cutoff_demo.py
Output:
    lowpass_cutoff_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "lowpass_cutoff_demo.gif")

ACTIVE_COLOR = "#e07b00"
NEUTRAL_COLOR = "#222222"

photo = np.zeros((200, 200), dtype=np.float64)
cv2.rectangle(photo, (40, 40), (120, 120), 200, -1)
cv2.circle(photo, (150, 150), 30, 120, -1)


def circular_mask(shape, radius):
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    return dist <= radius


def apply_lowpass(image, radius):
    F = np.fft.fftshift(np.fft.fft2(image))
    mask = circular_mask(image.shape, radius)
    filtered = np.fft.ifft2(np.fft.ifftshift(F * mask))
    return np.real(filtered), mask


def render(radius):
    filtered, mask = apply_lowpass(photo, radius)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), dpi=120)
    fig.subplots_adjust(top=0.86, bottom=0.05, wspace=0.15)

    axes[0].imshow(mask, cmap="gray")
    axes[0].set_title(f"low-pass mask, radius={radius:.0f}", fontsize=10, color=ACTIVE_COLOR)
    axes[0].axis("off")

    axes[1].imshow(filtered, cmap="gray")
    axes[1].set_title("filtered image", fontsize=10, color=NEUTRAL_COLOR)
    axes[1].axis("off")

    fig.suptitle("varying the ideal low-pass cutoff radius", fontsize=12.5,
                 color=NEUTRAL_COLOR, fontweight="bold", y=0.98)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


shrink = np.concatenate([
    np.linspace(90, 8, 50),
    np.full(20, 8),
])
radii = np.concatenate([shrink, shrink[::-1]])

frames = [render(r) for r in radii]

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=80,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
