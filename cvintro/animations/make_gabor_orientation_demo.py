"""Generate gabor_orientation_demo.gif: an animated companion to the lesson 16
notebook's "Orientation selectivity, demonstrated quantitatively" cells. A
Gabor filter's tuning angle theta sweeps through 0-180 degrees; at each
instant the filter is applied to the four-bars test image (0, 45, 90, 135
degrees) and the per-bar response magnitude is shown as a live bar chart.
The bar whose orientation currently matches the filter's tuning lights up
in ACTIVE_COLOR on both the image and the chart, visually demonstrating the
same diagonal-dominant response matrix the notebook computes numerically.

Run from anywhere:
    python make_gabor_orientation_demo.py
Output:
    gabor_orientation_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "gabor_orientation_demo.gif")

ACTIVE_COLOR = "#e07b00"
NEUTRAL_COLOR = "#222222"
DIM_COLOR = "#9a9a9a"


def draw_bar(image, center, angle_deg, length=60, thickness=6, value=255):
    angle = np.radians(angle_deg)
    dx, dy = length / 2 * np.cos(angle), length / 2 * np.sin(angle)
    p1 = (int(center[0] - dx), int(center[1] - dy))
    p2 = (int(center[0] + dx), int(center[1] + dy))
    cv2.line(image, p1, p2, value, thickness)


bar_orientations = [0, 45, 90, 135]
centers = [(50, 50), (150, 50), (50, 150), (150, 150)]

bars_img = np.zeros((200, 200), dtype=np.float64)
for c, ang in zip(centers, bar_orientations):
    draw_bar(bars_img, c, ang)


def responses_for_theta(theta_deg):
    # Same +90 offset convention as the notebook: cv2's theta is the
    # orientation of the stripes *inside* the kernel, which runs
    # perpendicular to the bar it responds to.
    kernel = cv2.getGaborKernel((25, 25), sigma=4, theta=np.radians(theta_deg + 90),
                                  lambd=10, gamma=0.5, psi=0)
    response = cv2.filter2D(bars_img, cv2.CV_64F, kernel)
    vals = np.zeros(4)
    for ci, c in enumerate(centers):
        region = response[c[1] - 20:c[1] + 20, c[0] - 20:c[0] + 20]
        vals[ci] = np.abs(region).mean()
    return kernel, vals


# normalize bar-chart heights against the max response seen across the full sweep
_sweep_theta = np.linspace(0, 180, 60, endpoint=False)
_all_vals = np.stack([responses_for_theta(t)[1] for t in _sweep_theta])
MAX_RESPONSE = _all_vals.max()


def render(theta_deg):
    kernel, vals = responses_for_theta(theta_deg)
    closest_idx = int(np.argmin([min(abs(theta_deg - a), abs(theta_deg - a - 180), abs(theta_deg - a + 180))
                                   for a in bar_orientations]))

    fig = plt.figure(figsize=(11.0, 5.2), dpi=120)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.45, 0.85], left=0.04, right=0.97,
                            top=0.80, bottom=0.1, wspace=0.3)
    ax_img = fig.add_subplot(gs[0])
    ax_kernel = fig.add_subplot(gs[1])
    ax_bar = fig.add_subplot(gs[2])

    ax_img.imshow(bars_img, cmap="gray", vmin=0, vmax=255)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for ci, (c, ang) in enumerate(zip(centers, bar_orientations)):
        is_match = ci == closest_idx
        color = ACTIVE_COLOR if is_match else DIM_COLOR
        lw = 2.4 if is_match else 1.0
        ax_img.add_patch(Rectangle((c[0] - 28, c[1] - 28), 56, 56, fill=False,
                                     edgecolor=color, linewidth=lw))
        ax_img.text(c[0], c[1] + 42, f"{ang}°", ha="center", va="top", fontsize=9, color=color)
    ax_img.set_title("four-bars test image", fontsize=10.5)

    ax_kernel.imshow(kernel, cmap="gray")
    ax_kernel.set_xticks([])
    ax_kernel.set_yticks([])
    ax_kernel.set_title(f"Gabor kernel\n" + r"$\theta$" + f" = {theta_deg:.0f}°", fontsize=10, color=ACTIVE_COLOR)

    bar_colors = [ACTIVE_COLOR if i == closest_idx else DIM_COLOR for i in range(4)]
    ax_bar.bar([str(a) + "°" for a in bar_orientations], vals, color=bar_colors)
    ax_bar.set_ylim(0, MAX_RESPONSE * 1.15)
    ax_bar.set_ylabel("response magnitude", fontsize=9.5)
    ax_bar.set_xlabel("bar orientation", fontsize=9.5)
    ax_bar.set_title("per-bar response", fontsize=10.5)
    for spine in ["top", "right"]:
        ax_bar.spines[spine].set_visible(False)

    fig.suptitle("Gabor filter tuning sweeps across orientation", fontsize=12.5,
                   color=NEUTRAL_COLOR, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


N = 72
thetas = np.linspace(0, 180, N, endpoint=False)

frames = [render(t) for t in thetas]

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=80,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
