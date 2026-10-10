"""Generate rgb_hsv_cube_demo.gif: an animated companion to the lesson 18
notebook's HSV intro cell. Shows a single color as a point moving
simultaneously in two 3D spaces: the RGB cube and the HSV cylinder. First the
point sweeps through hue at full saturation/value (tracing the outer rim of
both the cube and the cylinder's top rim), then it dims straight down
(value falling at fixed hue/saturation) -- demonstrating that "getting
darker" is a straight vertical drop in HSV but a diagonal move toward the
origin in RGB.

Run from anywhere:
    python make_rgb_hsv_cube_demo.py
Output:
    rgb_hsv_cube_demo.gif  (written next to this script)
"""
import colorsys
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "rgb_hsv_cube_demo.gif")

NEUTRAL_COLOR = "#222222"

ELEV, AZIM = 22, -50


def draw_rgb_cube(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    # keep the tick *positions* (0, 1) so mplot3d's default label placement
    # still has something to offset from, but hide the tick *numbers* --
    # otherwise every corner would show up to 3 overlapping "0"/"1" labels
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_zticks([0, 1])
    ax.set_xticklabels(["", ""]); ax.set_yticklabels(["", ""]); ax.set_zticklabels(["", ""])
    ax.set_xlabel("R", fontsize=9, labelpad=-12)
    ax.set_ylabel("G", fontsize=9, labelpad=-12)
    ax.set_zlabel("B", fontsize=9, labelpad=-12)
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
    edges = [(a, b) for i, a in enumerate(corners) for b in corners[i + 1:]
             if np.sum(np.abs(a - b)) == 1]
    for a, b in edges:
        ax.plot(*zip(a, b), color="#bbbbbb", linewidth=0.8, zorder=1)
    # each vertex is a pure/mixed primary -- its coordinates ARE its color
    ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], s=110, c=corners,
                edgecolors="#888888", linewidths=0.6, zorder=3)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_title("RGB cube", fontsize=10.5)


def draw_hsv_cylinder(ax):
    theta = np.linspace(0, 2 * np.pi, 48)
    ax.plot(np.cos(theta), np.sin(theta), 0.0, color="#bbbbbb", linewidth=0.8, zorder=1)
    # the top rim is exactly the s=1, v=1 ring -- color it by hue so it
    # reads as a rainbow, instead of a plain gray circle
    rim_theta = np.linspace(0, 2 * np.pi, 120)
    rim_colors = [colorsys.hsv_to_rgb(t / (2 * np.pi), 1.0, 1.0) for t in rim_theta]
    ax.scatter(np.cos(rim_theta), np.sin(rim_theta), 1.0, c=rim_colors, s=10, zorder=1)
    for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ax.plot([np.cos(a), np.cos(a)], [np.sin(a), np.sin(a)], [0, 1], color="#dddddd", linewidth=0.6, zorder=1)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 1)
    ax.set_xlabel("", fontsize=9); ax.set_ylabel("", fontsize=9)
    ax.set_zlabel("value", fontsize=9, labelpad=-12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_zticks([0, 1])
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_title("HSV cylinder", fontsize=10.5)


def render(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)

    fig = plt.figure(figsize=(9.2, 5.2), dpi=120)
    ax_rgb = fig.add_subplot(1, 2, 1, projection="3d")
    ax_hsv = fig.add_subplot(1, 2, 2, projection="3d")

    draw_rgb_cube(ax_rgb)
    ax_rgb.scatter([r], [g], [b], s=90, color=(r, g, b), edgecolors="black", linewidths=1.6, zorder=5)

    draw_hsv_cylinder(ax_hsv)
    x, y = s * np.cos(h * 2 * np.pi), s * np.sin(h * 2 * np.pi)
    ax_hsv.scatter([x], [y], [v], s=90, color=(r, g, b), edgecolors="black", linewidths=1.6, zorder=5)

    fig.suptitle(f"RGB=({r:.2f}, {g:.2f}, {b:.2f})    HSV=({h:.2f}, {s:.2f}, {v:.2f})",
                 fontsize=11, color=NEUTRAL_COLOR)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04, wspace=0.05)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []

N_HUE = 48
for h in np.linspace(0, 1, N_HUE, endpoint=False):
    frames.append(render(h, 1.0, 1.0))

N_VAL = 32
for v in np.linspace(1.0, 0.0, N_VAL):
    frames.append(render(0.0, 1.0, v))

frames += [frames[-1]] * 14

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=70,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
