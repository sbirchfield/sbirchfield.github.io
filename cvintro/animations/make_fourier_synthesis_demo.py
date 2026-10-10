"""Generate fourier_synthesis_demo.gif: an animated version of the lesson 15
notebook's 1D square-wave-from-sines cell. Builds up the partial Fourier sum
one odd harmonic at a time (1, 3, 5, ..., 19 terms), showing the running
approximation converge toward the true square wave while Gibbs
overshoot/ringing near the jumps never fully goes away.

Run from anywhere:
    python make_fourier_synthesis_demo.py
Output:
    fourier_synthesis_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "fourier_synthesis_demo.gif")

ACTIVE_COLOR = "#e07b00"
NEUTRAL_COLOR = "#222222"

t = np.linspace(0, 1, 500, endpoint=False)
square_wave = np.sign(np.sin(2 * np.pi * 5 * t))

MAX_HARMONICS = 19  # k = 1, 3, 5, ..., 19 -> 10 terms


def partial_sum(n_harmonics):
    return sum((4 / (np.pi * k)) * np.sin(2 * np.pi * 5 * k * t) for k in range(1, n_harmonics + 1, 2))


def render(n_harmonics):
    approx = partial_sum(n_harmonics)
    n_terms = (n_harmonics + 1) // 2

    fig, ax = plt.subplots(figsize=(7.5, 4.3), dpi=120)
    fig.subplots_adjust(top=0.82, bottom=0.12)
    ax.plot(t, square_wave, "--", color="#999999", linewidth=1.2, label="true square wave")
    ax.plot(t, approx, color=ACTIVE_COLOR, linewidth=1.8, label=f"{n_terms} harmonic(s)")
    ax.set_ylim(-1.55, 1.55)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlabel("t", fontsize=10, color=NEUTRAL_COLOR)
    fig.suptitle("building a square wave from sines", fontsize=12.5,
                 color=NEUTRAL_COLOR, fontweight="bold")
    ax.set_title(f"{n_harmonics} terms (odd harmonics up to k={n_harmonics})", fontsize=10, color=NEUTRAL_COLOR)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


harmonics_sequence = list(range(1, MAX_HARMONICS + 1, 2))

frames = []
HOLD = 10
HOLD_END = 45
for n in harmonics_sequence:
    frames += [render(n)] * HOLD
frames += [render(harmonics_sequence[-1])] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=90,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
