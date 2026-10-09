"""Generate blend_demo.gif: a continuous cross-dissolve between two images,
sweeping alpha from 0 to 1 and back, with a slider showing where in that
range the current frame sits.

Uses the same images and formula as the lesson 2 notebook's "Blending two
images" cell: cv2.addWeighted(img_a, 1 - alpha, img_b, alpha, 0).

Run from anywhere:
    python make_blend_demo.py
Output:
    blend_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "blend_demo.gif")
IMG_DIR = os.path.join(OUT_DIR, "..", "img")

ACTIVE_COLOR = "#e07b00"

# Same images and resize-to-match step as the lesson 2 notebook's blending cell.
img_a = cv2.imread(os.path.join(IMG_DIR, "sheepdog.jpg"))
img_b = cv2.imread(os.path.join(IMG_DIR, "lhasa_apso.jpg"))
img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))

img_a_rgb = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
img_b_rgb = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)


SLIDER_TICKS = [0.0, 0.25, 0.5, 0.75, 1.0]
SLIDER_LABELS = ["0", "0.25", "0.5", "0.75", "1.0"]


def render(alpha):
    blended = cv2.addWeighted(img_a, 1 - alpha, img_b, alpha, 0)
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(6.0, 6.3), dpi=120)
    gs = fig.add_gridspec(2, 3, height_ratios=[4.0, 0.6], width_ratios=[0.16, 0.68, 0.16],
                            left=0.04, right=0.96, top=0.88, bottom=0.08, hspace=0.12, wspace=0.1)
    ax_main = fig.add_subplot(gs[0, :])
    ax_thumb_a = fig.add_subplot(gs[1, 0])
    ax_slider = fig.add_subplot(gs[1, 1])
    ax_thumb_b = fig.add_subplot(gs[1, 2])

    ax_main.imshow(blended_rgb)
    ax_main.axis("off")
    ax_main.set_title("cross-dissolve", fontsize=13, color="#333333")

    ax_thumb_a.imshow(img_a_rgb)
    ax_thumb_a.axis("off")

    ax_thumb_b.imshow(img_b_rgb)
    ax_thumb_b.axis("off")

    ax_slider.axis("off")
    ax_slider.set_xlim(-0.08, 1.08)
    ax_slider.set_ylim(-2.1, 1)
    ax_slider.plot([0, 1], [0, 0], color="#cccccc", lw=4, solid_capstyle="round", zorder=1)
    for t, label in zip(SLIDER_TICKS, SLIDER_LABELS):
        ax_slider.plot([t, t], [-0.22, 0.22], color="#999999", lw=1.2, zorder=1)
        ax_slider.text(t, -0.55, label, ha="center", va="top", fontsize=7.5, color="#555555")
    ax_slider.text(0.5, -1.3, "alpha", ha="center", va="top", fontsize=9, color="#333333")
    ax_slider.scatter([alpha], [0], s=140, color=ACTIVE_COLOR, zorder=2,
                        edgecolors="white", linewidths=1.5)

    fig.text(0.5, 0.965, "cv2.addWeighted(img_a, 1 - alpha, img_b, alpha, 0)",
               ha="center", va="top", fontsize=9.5, color="#555555")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


N = 36  # steps from alpha=0 to alpha=1
HOLD_END = 8

alphas_fwd = np.linspace(0.0, 1.0, N + 1)
alphas_bwd = alphas_fwd[::-1][1:-1]  # avoid repeating the endpoints

frames = []
frames += [render(0.0)] * HOLD_END
for a in alphas_fwd[1:]:
    frames.append(render(a))
frames += [render(1.0)] * HOLD_END
for a in alphas_bwd:
    frames.append(render(a))

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=70,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
