"""Generate separable_pass_demo.gif: an animated walkthrough of applying a
separable kernel as two 1D passes -- convolve each row with kx, then
convolve each column of that with ky -- compared against convolving with
the full 2D kernel directly.

Uses the same image and kernel values as the lesson 10 notebook's
separable-kernels cell.

Run from anywhere:
    python make_separable_pass_demo.py
Output:
    separable_pass_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "separable_pass_demo.gif")

# Same 1D Gaussian kernel as the lesson 10 notebook's separable-kernels cell.
sigma = 1.0
k1d = np.exp(-(np.arange(5) - 2) ** 2 / (2 * sigma ** 2))
k1d /= k1d.sum()
K = len(k1d)
PAD = K // 2
gaussian2d = np.outer(k1d, k1d)

# Same small "bump" image used in the other 2D convolution demos.
image = np.array([
    [1, 2, 3, 2, 1],
    [2, 3, 4, 3, 2],
    [3, 4, 5, 4, 3],
    [2, 3, 4, 3, 2],
    [1, 2, 3, 2, 1],
], dtype=np.float64)
H, W = image.shape


def conv1d(signal, kernel):
    pad = len(kernel) // 2
    padded = np.pad(signal, pad, mode="constant")
    flipped = kernel[::-1]  # symmetric here, but keep the real definition
    out = np.zeros_like(signal, dtype=np.float64)
    for n in range(len(signal)):
        out[n] = np.dot(padded[n:n + len(kernel)], flipped)
    return out


row_pass = np.array([conv1d(image[r, :], k1d) for r in range(H)])             # convolve each row
final_separable = np.array([conv1d(row_pass[:, c], k1d) for c in range(W)]).T  # then each column

padded_img = np.pad(image, PAD, mode="constant")
flipped2d = gaussian2d[::-1, ::-1]
direct_2d = np.array([
    [float(np.sum(padded_img[r:r + K, c:c + K] * flipped2d)) for c in range(W)]
    for r in range(H)
])

assert np.allclose(final_separable, direct_2d), "separable result does not match direct 2D convolution!"

ACTIVE_COLOR = "#e07b00"
DONE_COLOR = "#3a6fb0"
MATCH_COLOR = "#2e9e44"
CELL_FACE = "#ffffff"


def fmt(v):
    return "0" if abs(v) < 1e-9 else f"{v:.2f}".rstrip("0").rstrip(".")


def draw_image_panel(ax, M, title, face=CELL_FACE, text_color="#222222", show=True):
    for r in range(H):
        for c in range(W):
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=face if show else "#f2f2f2",
                                     edgecolor="#aaaaaa", linewidth=0.7, zorder=0))
            if show:
                ax.text(c, r, fmt(M[r, c]), ha="center", va="center", fontsize=9,
                         color=text_color, fontweight="bold", zorder=1)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)


def render(stage):
    """stage: 0 = image only, 1 = + row pass, 2 = + column pass,
    3 = + direct 2D result (final hold, shows the match)."""
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.8), dpi=120,
                               gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    fig.subplots_adjust(left=0.03, right=0.98, top=0.72, bottom=0.08, wspace=0.35)
    ax_img, ax_mid, ax_final, ax_direct = axes

    draw_image_panel(ax_img, image, "image")
    draw_image_panel(ax_mid, row_pass, "after row pass\n(⊛ kx)",
                       face="#fff1e0" if stage == 1 else CELL_FACE,
                       text_color=ACTIVE_COLOR if stage == 1 else DONE_COLOR, show=stage >= 1)
    draw_image_panel(ax_final, final_separable, "after column pass\n(⊛ ky)",
                       face="#fff1e0" if stage == 2 else CELL_FACE,
                       text_color=ACTIVE_COLOR if stage == 2 else DONE_COLOR, show=stage >= 2)
    draw_image_panel(ax_direct, direct_2d, "direct 2D conv.\n(⊛ K)",
                       face="#eafaf0" if stage >= 3 else CELL_FACE,
                       text_color=MATCH_COLOR if stage >= 3 else "#222222", show=stage >= 3)

    titles = {
        0: "start with the image",
        1: "pass 1: convolve each row with kx (1×5)",
        2: "pass 2: convolve each column of that with ky (5×1)",
        3: "compare to convolving with the full 5×5 kernel directly — identical",
    }
    color = MATCH_COLOR if stage >= 3 else ACTIVE_COLOR
    fig.suptitle(titles[stage], fontsize=13, color=color)

    if stage >= 3:
        ax_final.text(0.5, 1.26, "=", transform=ax_final.transAxes, ha="center", va="bottom",
                        fontsize=18, color=MATCH_COLOR, fontweight="bold")
        ax_direct.text(0.5, 1.26, "MATCH", transform=ax_direct.transAxes, ha="center", va="bottom",
                         fontsize=14, color=MATCH_COLOR, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STAGE = 28
HOLD_END = 40

for stage in range(3):
    frames += [render(stage)] * HOLD_STAGE
frames += [render(3)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=110,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
