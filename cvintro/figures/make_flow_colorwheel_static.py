"""Generate flow_colorwheel.png: a static legend for the dense-flow color
coding used throughout the Farneback section (flow_to_color in the
notebook). Built from the exact same HSV convention -- hue = angle*180/pi/2
(OpenCV's 0-180 hue range), saturation = 255, value = magnitude -- so the
legend's colors are pixel-identical to what the flow visualizations actually
show, not a stylized stand-in. Radius from the center encodes speed
(magnitude): the center is dark/slow, the rim is bright/fast.

Run from anywhere:
    python make_flow_colorwheel.py
Output:
    flow_colorwheel.png  (written next to this script)
"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(OUT_DIR, "flow_colorwheel.png")

NEUTRAL_COLOR = "#222222"

N = 400
yy, xx = np.mgrid[0:N, 0:N]
cx, cy = N / 2, N / 2
u = (xx - cx)
v = (yy - cy)  # image-style y-down, matches flow fields' v component

flow = np.stack([u, v], axis=-1).astype(np.float32)
magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
radius = np.hypot(xx - cx, yy - cy)
mask = radius <= N / 2

hsv = np.zeros((N, N, 3), dtype=np.uint8)
hsv[..., 0] = angle * 180 / np.pi / 2
hsv[..., 1] = 255
hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
wheel = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
wheel[~mask] = 255

fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=130)
ax.imshow(wheel)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

arrow_r = N / 2 - 45
labels = [("right", 0), ("down", 90), ("left", 180), ("up", 270)]
for text, deg in labels:
    rad = np.radians(deg)
    x = cx + arrow_r * np.cos(rad)
    y = cy + arrow_r * np.sin(rad)
    ax.annotate(text, xy=(x, y), ha="center", va="center", fontsize=9.5,
                color="white", fontweight="bold")

ax.annotate("slow\n(center)", xy=(cx, cy), ha="center", va="center", fontsize=8,
            color="white", fontweight="bold")
ax.set_title("hue = direction, brightness = speed", fontsize=11, color=NEUTRAL_COLOR)

plt.tight_layout()
fig.savefig(OUT_PNG)
plt.close(fig)
print(f"Saved {OUT_PNG}")
