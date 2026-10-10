"""Generate grating_spectrum_demo.gif: an animated version of the lesson 15
notebook's grating-spectrum-peak cell. A sinusoidal grating image rotates
through orientations while its FFT log-magnitude spectrum is shown side by
side -- the bright peak pair rotates with it, always in the direction
perpendicular to the stripes.

Run from anywhere:
    python make_grating_spectrum_demo.py
Output:
    grating_spectrum_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "grating_spectrum_demo.gif")

ACTIVE_COLOR = "#e07b00"
NEUTRAL_COLOR = "#222222"

SIZE = 100
CYCLES_ACROSS_IMAGE = 12
yy, xx = np.mgrid[:SIZE, :SIZE]
cyy, cxx = yy - SIZE / 2, xx - SIZE / 2


def make_grating(theta_deg):
    theta = np.deg2rad(theta_deg)
    proj = cxx * np.cos(theta) + cyy * np.sin(theta)
    return np.sin(2 * np.pi * CYCLES_ACROSS_IMAGE * proj / SIZE)


def spectrum(image):
    F = np.fft.fftshift(np.fft.fft2(image))
    return np.log1p(np.abs(F))


def render(theta_deg):
    grating = make_grating(theta_deg)
    mag = spectrum(grating)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6), dpi=120)
    fig.subplots_adjust(top=0.86, bottom=0.05, wspace=0.15)
    axes[0].imshow(grating, cmap="gray")
    axes[0].set_title(f"grating, stripes at {theta_deg:.0f}°", fontsize=10, color=ACTIVE_COLOR)
    axes[0].axis("off")

    axes[1].imshow(mag, cmap="gray")
    axes[1].set_title("spectrum (peak pair rotates too)", fontsize=10, color=NEUTRAL_COLOR)
    axes[1].axis("off")

    fig.suptitle("varying the grating orientation", fontsize=12.5,
                 color=NEUTRAL_COLOR, fontweight="bold", y=0.98)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


N = 72
thetas = np.linspace(0, 180, N, endpoint=False)

frames = [render(theta) for theta in thetas]

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=320,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
