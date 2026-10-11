"""Generate defocus_blur_demo.gif: an animated version of the lesson 26
notebook's "Real lenses: focus and defocus blur" cell. Uses the notebook's
own 3-block synthetic scene (near/mid/far objects) and defocus_kernel_size()
formula, but sweeps the camera's focus distance continuously from near to
far and back, instead of fixing it at 5m -- showing the circle of confusion
(blur kernel size) grow and shrink on each block live as the focal plane
sweeps through it.

Run from anywhere:
    python make_defocus_blur_demo.py
Output:
    defocus_blur_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "defocus_blur_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"

# Same synthetic scene as the notebook cell.
scene = np.zeros((200, 300, 3), dtype=np.uint8)
cv2.rectangle(scene, (20, 20), (90, 180), (255, 120, 30), -1)    # near object
cv2.rectangle(scene, (110, 20), (190, 180), (30, 200, 255), -1)  # object at the focal plane
cv2.rectangle(scene, (210, 20), (280, 180), (120, 255, 60), -1)  # far object

# A letter on each block, drawn into the scene itself (so it blurs along with
# the block) gives the eye something with fine detail to judge sharpness by.
_FONT = cv2.FONT_HERSHEY_SIMPLEX
for cx, cy in [(55, 100), (150, 100), (245, 100)]:
    (tw, th), _ = cv2.getTextSize("E", _FONT, 2.6, 6)
    org = (cx - tw // 2, cy + th // 2)
    cv2.putText(scene, "E", org, _FONT, 2.6, (0, 0, 0), 6, cv2.LINE_AA)

depths_m = {"near": 2.0, "mid": 5.0, "far": 9.0}
REGIONS = {"near": (20, 180, 20, 90), "mid": (20, 180, 110, 190), "far": (20, 180, 210, 280)}
# Same colors as the blocks themselves (BGR tuples above, converted to RGB 0-1).
DOT_COLORS = {"near": (30 / 255, 120 / 255, 255 / 255),
              "mid": (255 / 255, 200 / 255, 30 / 255),
              "far": (60 / 255, 255 / 255, 120 / 255)}
SLIDER_MIN, SLIDER_MAX = 0.0, 10.5


def defocus_kernel_size(depth, focus, strength=15.0):
    radius = strength * abs(depth - focus) / focus
    ksize = int(radius) * 2 + 1
    return max(1, ksize)


def blur_with_context(image, y0, y1, x0, x1, ksize, pad):
    if ksize <= 1:
        return image[y0:y1, x0:x1].copy()
    yy0, yy1 = max(0, y0 - pad), min(image.shape[0], y1 + pad)
    xx0, xx1 = max(0, x0 - pad), min(image.shape[1], x1 + pad)
    blurred = cv2.GaussianBlur(image[yy0:yy1, xx0:xx1], (ksize, ksize), 0)
    return blurred[y0 - yy0:y0 - yy0 + (y1 - y0), x0 - xx0:x0 - xx0 + (x1 - x0)]


def render(focus_depth_m):
    defocused = scene.copy()
    ksizes = {}
    for name, (y0, y1, x0, x1) in REGIONS.items():
        k = defocus_kernel_size(depths_m[name], focus_depth_m)
        ksizes[name] = k
        defocused[y0:y1, x0:x1] = blur_with_context(scene, y0, y1, x0, x1, k, pad=k)

    # Whichever object currently has the smallest kernel is the one in focus.
    in_focus_name = min(ksizes, key=ksizes.get)

    fig = plt.figure(figsize=(7.2, 4.0), dpi=120)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 3.0], top=0.73, bottom=0.05,
                            left=0.05, right=0.95, hspace=0.75)
    ax_slider = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])

    ax.imshow(cv2.cvtColor(defocused, cv2.COLOR_BGR2RGB))
    ax.axis("off")

    # Horizontal slider: a fixed dot per object (positioned at its true depth,
    # colored to match the block) plus a moving marker for the current focus
    # distance -- whichever dot the marker is closest to is the sharp one.
    ax_slider.axhline(0, color="#cccccc", linewidth=2, zorder=1)
    for name, depth in depths_m.items():
        is_focus = name == in_focus_name
        ax_slider.scatter([depth], [0], s=260 if is_focus else 180, color=DOT_COLORS[name],
                            edgecolors=MATCH_COLOR if is_focus else "white",
                            linewidths=2.5 if is_focus else 1.5, zorder=3)
    ax_slider.plot([focus_depth_m], [0], marker="v", markersize=14, color=ACTIVE_COLOR,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=4,
                    clip_on=False)
    ax_slider.text(focus_depth_m, 0.75, f"{focus_depth_m:4.1f} m", ha="center", va="bottom",
                     fontsize=10.5, color=ACTIVE_COLOR, fontweight="bold", family="monospace")
    ax_slider.set_xlim(SLIDER_MIN, SLIDER_MAX)
    ax_slider.set_ylim(-1.0, 1.4)
    ax_slider.set_xticks([0, 2, 5, 9])
    ax_slider.set_xticklabels(["0 m", "near (2m)", "mid (5m)", "far (9m)"], fontsize=8.5)
    ax_slider.tick_params(axis="x", length=0, pad=6)
    ax_slider.set_yticks([])
    for spine in ("left", "right", "top", "bottom"):
        ax_slider.spines[spine].set_visible(False)
    ax_slider.set_xlabel("focus distance", fontsize=9, color=NEUTRAL_COLOR, labelpad=2)

    fig.suptitle(f"focused at {focus_depth_m:4.1f} m", fontsize=13, color=ACTIVE_COLOR,
                  fontweight="bold", y=0.82)
    fig.text(0.5, 0.49, "circle-of-confusion size grows with distance from the focal plane",
              ha="center", va="top", fontsize=9.5, color=NEUTRAL_COLOR)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


sweep = np.concatenate([
    np.linspace(1.0, 9.5, 36),
    np.linspace(9.5, 1.0, 36),
])

frames = [render(f) for f in sweep]

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=80,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
