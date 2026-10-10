"""Generate frequency_spectrum_demo.gif: a companion to grating_spectrum_demo.gif
for the lesson 15 notebook's grating-spectrum-peak cell. A sinusoidal grating
image at a fixed orientation has its frequency (cycles across the image)
swept from low to high and back -- its FFT log-magnitude spectrum is shown
side by side, with the bright peak pair moving outward from the center as
the frequency increases, and back inward as it decreases.

Run from anywhere:
    python make_frequency_spectrum_demo.py
Output:
    frequency_spectrum_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "frequency_spectrum_demo.gif")

ACTIVE_COLOR = "#e07b00"
NEUTRAL_COLOR = "#222222"

SIZE = 100
THETA_DEG = 30
yy, xx = np.mgrid[:SIZE, :SIZE]
cyy, cxx = yy - SIZE / 2, xx - SIZE / 2
theta = np.deg2rad(THETA_DEG)
proj = cxx * np.cos(theta) + cyy * np.sin(theta)


def make_grating(cycles):
    return np.sin(2 * np.pi * cycles * proj / SIZE)


def spectrum(image):
    F = np.fft.fftshift(np.fft.fft2(image))
    return np.log1p(np.abs(F))


def render(cycles):
    grating = make_grating(cycles)
    mag = spectrum(grating)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6), dpi=120)
    fig.subplots_adjust(top=0.86, bottom=0.05, wspace=0.15)
    axes[0].imshow(grating, cmap="gray")
    axes[0].set_title(f"grating, {cycles:.1f} cycles across image", fontsize=10, color=ACTIVE_COLOR)
    axes[0].axis("off")

    axes[1].imshow(mag, cmap="gray")
    axes[1].set_title("spectrum (peak pair moves out/in)", fontsize=10, color=NEUTRAL_COLOR)
    axes[1].axis("off")

    fig.suptitle("varying the grating frequency", fontsize=12.5,
                 color=NEUTRAL_COLOR, fontweight="bold", y=0.98)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


rise = np.concatenate([
    np.linspace(1, 24, 50),
    np.full(20, 24),
])
cycles_seq = np.concatenate([rise, rise[::-1]])

frames = [render(c) for c in cycles_seq]

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=160,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
