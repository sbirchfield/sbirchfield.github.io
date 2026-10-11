"""Generate aperture_problem_demo.gif: an animated version of the lesson 21
notebook's aperture-problem cell. Reuses that cell's exact rectangle setup.
Simple, literal version: the white rectangle's edge actually moves diagonally
(both along-edge and perpendicular components are nonzero and changing), and
a fixed aperture window straddles the edge. Side by side with the full image,
the window's cropped content is shown -- it only ever shifts vertically
(perpendicular to the edge), never horizontally, even though the true motion
is diagonal. That's the aperture problem, directly visible rather than
asserted.

Run from anywhere:
    python make_aperture_problem_demo.py
Output:
    aperture_problem_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "aperture_problem_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"

# Same setup as the notebook's aperture-problem cell.
size = 160
rect_topleft = (50, 60)
rect_w, rect_h = 60, 40


def make_rect_frame(dx=0, dy=0, value=200):
    img = np.zeros((size, size), dtype=np.uint8)
    x0, y0 = rect_topleft[0] + dx, rect_topleft[1] + dy
    cv2.rectangle(img, (x0, y0), (x0 + rect_w, y0 + rect_h), value, -1)
    return img


# A small aperture window straddling the rectangle's TOP (horizontal) edge,
# fixed in image coordinates -- centered well away from any corner, and the
# diagonal sweep below stays small enough that the edge still spans the
# window's full width at every frame.
win_half = 10
wx, wy = rect_topleft[0] + rect_w // 2, rect_topleft[1]

# True motion is diagonal: both dx (along-edge) and dy (perpendicular) move
# together, back and forth along a straight diagonal line. Equal amplitude in
# x and y, with step count chosen so each integer-pixel position changes by
# exactly 1 every frame in BOTH dx and dy -- unequal amplitudes (e.g. twice
# as much horizontal range as vertical) make dy round to the same integer
# for two frames in a row while dx keeps advancing, which looks like
# "right, right, down, right, right, down" instead of a smooth diagonal.
AMP = 10
N_STEPS = 21  # linspace(-1, 1, 21) has step 0.1, so AMP * step = 1 exactly
m = np.concatenate([np.linspace(-1, 1, N_STEPS), np.linspace(1, -1, N_STEPS)])
DX = AMP * m
DY = AMP * m

# Sanity check (matches the notebook's own assertion): the window crop only
# depends on dy, not dx -- a horizontal shift of a horizontal edge leaves the
# local appearance unchanged as long as the edge still spans the window.
for dy in [-AMP, 0, AMP]:
    crop0 = make_rect_frame(int(round(DX[0])), dy)[wy - win_half:wy + win_half, wx - win_half:wx + win_half]
    for dx in DX:
        crop = make_rect_frame(int(round(dx)), dy)[wy - win_half:wy + win_half, wx - win_half:wx + win_half]
        assert np.array_equal(crop, crop0)


def render(dx, dy):
    frame = make_rect_frame(int(round(dx)), int(round(dy)))
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    crop = vis[wy - win_half:wy + win_half, wx - win_half:wx + win_half]

    fig, (ax_main, ax_crop) = plt.subplots(
        1, 2, figsize=(8.2, 4.4), dpi=120, gridspec_kw={"width_ratios": [1.3, 1.0]},
    )
    fig.subplots_adjust(top=0.80, bottom=0.06, wspace=0.3)

    ax_main.imshow(vis)
    ax_main.add_patch(Rectangle((wx - win_half, wy - win_half), 2 * win_half, 2 * win_half,
                                  fill=False, edgecolor=ACTIVE_COLOR, linewidth=1.8))
    arrow_scale = 2.5
    tip = (wx + dx * arrow_scale, wy + dy * arrow_scale)
    # Near the direction reversal (dx, dy both ~0), matplotlib draws a
    # malformed arrowhead (a tiny blob, not a clean point) once the arrow
    # gets too short -- draw a plain line + dot there instead of an arrow,
    # rather than making the motion indicator disappear entirely.
    if np.hypot(dx, dy) > AMP / 2:
        ax_main.annotate("", xy=tip, xytext=(wx, wy),
                          arrowprops=dict(arrowstyle="->", color=ACTIVE_COLOR, lw=2.0))
    else:
        ax_main.plot([wx, tip[0]], [wy, tip[1]], color=ACTIVE_COLOR, linewidth=2.0)
        ax_main.scatter([tip[0]], [tip[1]], color=ACTIVE_COLOR, s=18, zorder=4)
    ax_main.set_title("true motion: diagonal", fontsize=10.5)
    ax_main.set_xticks([]); ax_main.set_yticks([])

    ax_crop.imshow(crop)
    ax_crop.set_title("window content:\nonly vertical motion visible", fontsize=10.5, color=NEUTRAL_COLOR)
    ax_crop.set_xticks([]); ax_crop.set_yticks([])
    for spine in ax_crop.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(ACTIVE_COLOR)
        spine.set_linewidth(2.0)

    fig.suptitle("The aperture problem", fontsize=12, color=NEUTRAL_COLOR, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 3
HOLD_END = 30

for dx, dy in zip(DX, DY):
    frames += [render(dx, dy)] * HOLD_STEP
frames += [render(DX[-1], DY[-1])] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=60,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
