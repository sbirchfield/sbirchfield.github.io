"""Generate bilinear_demo.gif: an animated version of the lesson 9 notebook's
"Bilinear interpolation" cell. Zooms into a single 2x2 pixel neighborhood
(straddling the triangle/background edge, so the four neighbors aren't all
the same color) and animates the fractional sample position (dx, dy) sweeping
around that 2x2 square, with each corner's weight and the resulting blended
color updating live.

Uses 4 real neighboring pixels from the same test image as the notebook.

Run from anywhere:
    python make_bilinear_demo.py
Output:
    bilinear_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "bilinear_demo.gif")

ACTIVE_COLOR = "#e07b00"


def make_test_image(size=60):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.rectangle(img, (5, 5), (size - 5, size - 5), (60, 90, 160), -1)
    tri = np.array([[30, 12], [10, 32], [50, 32]], dtype=np.int32)
    cv2.fillPoly(img, [tri], (225, 195, 60))
    return img


img = make_test_image()

# A 2x2 neighborhood straddling the triangle edge, so the 4 corners aren't
# all the same color -- makes the blend visually obvious.
X0, Y0 = 22, 19
I00 = img[Y0, X0] / 255.0       # top-left     (x0, y0)
I10 = img[Y0, X0 + 1] / 255.0   # top-right    (x1, y0)
I01 = img[Y0 + 1, X0] / 255.0   # bottom-left  (x0, y1)
I11 = img[Y0 + 1, X0 + 1] / 255.0  # bottom-right (x1, y1)

CORNERS = [
    ((0, 0), I00, "I(x0,y0)"),
    ((1, 0), I10, "I(x1,y0)"),
    ((0, 1), I01, "I(x0,y1)"),
    ((1, 1), I11, "I(x1,y1)"),
]


def weights(dx, dy):
    return [(1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy]


def blend(dx, dy):
    w = weights(dx, dy)
    colors = [I00, I10, I01, I11]
    return sum(wi * ci for wi, ci in zip(w, colors))


def render(dx, dy):
    w = weights(dx, dy)
    color = blend(dx, dy)

    fig = plt.figure(figsize=(9.0, 5.2), dpi=120)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.7], left=0.04, right=0.97,
                            top=0.84, bottom=0.08, wspace=0.18)
    ax = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])

    ax.set_aspect("equal")
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(1.35, -0.35)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["x0", "x1"])
    ax.set_yticklabels(["y0", "y1"])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for (cx, cy), color_i, label in CORNERS:
        ax.add_patch(Rectangle((cx - 0.5, cy - 0.5), 1, 1, facecolor=color_i,
                                 edgecolor="#999999", linewidth=0.8, zorder=1))

    for (cx, cy), color_i, label in CORNERS:
        ax.plot([dx, cx], [dy, cy], color="#555555", linewidth=0.8, zorder=2)
        ax.scatter([cx], [cy], s=40, color="white", edgecolors="#333333", linewidths=1.0, zorder=3)
        lx = cx + (0.22 if cx == 0 else -0.22)
        ly = cy + (0.14 if cy == 0 else -0.14)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=8, color="#222222",
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75))

    ax.scatter([dx], [dy], s=140, color=ACTIVE_COLOR, zorder=4, edgecolors="white", linewidths=1.4)
    ax.set_title("sample position (dx, dy)", fontsize=10.5)

    ax_info.axis("off")
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.text(0.0, 0.95, f"dx = {dx:.2f}   dy = {dy:.2f}", fontsize=13,
                   family="monospace", ha="left", va="top", color="#333333")
    labels = ["w00 = (1-dx)(1-dy)", "w10 = dx(1-dy)", "w01 = (1-dx)dy", "w11 = dx dy"]
    y = 0.82
    for label, wi in zip(labels, w):
        ax_info.text(0.0, y, f"{label:<18} = {wi:.2f}", fontsize=11.5, family="monospace",
                       ha="left", va="top", color="#333333")
        y -= 0.09
    ax_info.add_patch(Rectangle((0.0, y - 0.22), 0.3, 0.2, facecolor=color, edgecolor="#333333",
                                   linewidth=1.0, transform=ax_info.transAxes))
    ax_info.text(0.36, y - 0.12, "= blended color", fontsize=11.5, ha="left", va="center",
                   color="#333333")

    fig.suptitle("bilinear interpolation: blend the 4 neighbors by distance", fontsize=12.5,
                   color="#222222", fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


N = 80
t = np.linspace(0, 2 * np.pi, N, endpoint=False)
dxs = 0.5 + 0.47 * np.cos(t)
dys = 0.5 + 0.47 * np.sin(t)

frames = [render(dx, dy) for dx, dy in zip(dxs, dys)]

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=80,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
