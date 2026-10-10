"""Generate pyramid_demo.gif: an animated version of the lesson 11 notebook's
"Building a Gaussian pyramid" cell. Repeats "blur, then downsample" on a real
photo, with each finished level appended to a growing 3D "pancake stack" on
the side and a third column showing every level resized back up to a common
size, so the progressive loss of detail is obvious even for the tiny late
levels.

Run from anywhere:
    python make_pyramid_demo.py
Output:
    pyramid_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "pyramid_demo.gif")
IMG_DIR = os.path.join(OUT_DIR, "..", "img")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"
SIGMA = 1.0

photo_bgr = cv2.imread(os.path.join(IMG_DIR, "dog_looking.jpg"))
photo_bgr = cv2.resize(photo_bgr, (256, 256), interpolation=cv2.INTER_AREA)
photo = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)

NUM_LEVELS = 5
pyramid = [photo]
current = photo
blurred_levels = []
for _ in range(NUM_LEVELS - 1):
    b = cv2.GaussianBlur(current, (0, 0), sigmaX=SIGMA)
    blurred_levels.append(b)
    current = b[::2, ::2]
    pyramid.append(current)


# --- "pancake stack" rendering: each level drawn as a thin, sheared slab (as
# if viewed nearly edge-on from the side), stacked largest-at-bottom /
# smallest-at-top and centered on a common axis -- so the whole stack reads
# as a real (stepped) pyramid silhouette, upside down relative to the usual
# top-down diagram.
BASE_W = 1.0
FLATTEN = 0.45
SHEAR_FRAC = 0.45
OVERLAP = 0.12  # fraction of each slab's height hidden behind the one above it


def make_pancake(img, w_disp):
    """Returns an RGBA image (alpha=0 outside the sheared parallelogram, so
    the slab below shows through instead of a white border), plus its
    on-screen width (including the shear) and height. The destination
    quadrilateral is 2x as wide and half as tall as the "natural" px_w x px_h
    box, which is what actually flattens the pancake -- the canvas has to be
    widened to match (out_w) or the right side of the content gets clipped."""
    h, w = img.shape[:2]
    px_w = 240
    px_h = max(2, int(px_w * FLATTEN))
    shear_px = px_h * (SHEAR_FRAC / FLATTEN)
    src = np.float32([[0, 0], [w, 0], [0, h]])
    dst = np.float32([[shear_px, 0], [2 * px_w, 0], [0, 0.5 * px_h]])
    M = cv2.getAffineTransform(src, dst)
    out_w = int(np.ceil(2 * px_w + shear_px))
    warped = cv2.warpAffine(img, M, (out_w, px_h))
    mask = cv2.warpAffine(np.full((h, w), 255, dtype=np.uint8), M, (out_w, px_h))
    rgba = np.dstack([warped, mask])
    disp_h = w_disp * FLATTEN
    disp_shear = disp_h * (SHEAR_FRAC / FLATTEN)
    return rgba, w_disp + disp_shear, disp_h


def _pancake_layout(num_levels):
    """Precompute each level's (display width, height, y-position) once, for
    a fixed `num_levels`-tall stack -- shared by every frame so the stack
    doesn't subtly rescale/shift as new levels are added."""
    layout = []
    y = 0.0
    max_w = 0.0
    for i in range(num_levels):
        frac = pyramid[i].shape[0] / photo.shape[0]
        w_disp = BASE_W * frac
        disp_h = w_disp * FLATTEN
        disp_shear = disp_h * (SHEAR_FRAC / FLATTEN)
        disp_w = w_disp + disp_shear
        layout.append((w_disp, disp_w, disp_h, y))
        max_w = max(max_w, disp_w)
        y += disp_h * (1 - OVERLAP)
    total_h = y + (layout[-1][2] if layout else 0)
    return layout, max_w, total_h


PANCAKE_LAYOUT, PANCAKE_MAX_W, PANCAKE_TOTAL_H = _pancake_layout(NUM_LEVELS)


def draw_pancake_stack(ax_stack, done_levels):
    ax_stack.axis("off")
    cx = PANCAKE_MAX_W / 2

    # draw back-to-front (bottom/widest slab first) so each narrower slab
    # above it is layered on top, like a real stepped stack viewed from the side
    for i in range(done_levels):
        w_disp, disp_w, disp_h, y0 = PANCAKE_LAYOUT[i]
        img_rgba, _, _ = make_pancake(pyramid[i], w_disp)
        x0 = cx - disp_w / 2
        ax_stack.imshow(img_rgba, extent=[x0, x0 + disp_w, y0, y0 + disp_h], zorder=10 + i, aspect="auto")

    ax_stack.set_xlim(-0.05, PANCAKE_MAX_W + 0.05)
    ax_stack.set_ylim(-0.05, PANCAKE_TOTAL_H * 1.15)
    ax_stack.set_aspect("auto")


def draw_zoomed_stack(ax_zoom, done_levels):
    """Same stack, but every level is nearest-neighbor zoomed back up to the
    original size, so the shrinking levels occupy equal-sized slots -- making
    the progressive loss of detail obvious even for the tiny late levels."""
    ax_zoom.axis("off")
    ax_zoom.set_xlim(0, 1)
    ax_zoom.set_ylim(0, NUM_LEVELS)
    half = 0.42
    for i in range(done_levels):
        lvl = pyramid[i]
        zoomed = cv2.resize(lvl, (photo.shape[1], photo.shape[0]), interpolation=cv2.INTER_NEAREST)
        cx, cy = 0.5, i + 0.5
        ax_zoom.imshow(zoomed, extent=[cx - half, cx + half, cy - half, cy + half], zorder=2)


def render(stage, step_idx, final_hold=False):
    """stage: 0=blur, 1=downsample (level just added)."""
    done_levels = step_idx + (1 if stage == 1 else 0)
    cur_img = blurred_levels[step_idx] if stage == 0 else pyramid[step_idx + 1]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.4), dpi=120, gridspec_kw={"width_ratios": [1.0, 0.45, 0.45]})
    fig.subplots_adjust(top=0.78, bottom=0.08, wspace=0.25)
    ax_main, ax_stack, ax_zoom = axes

    ax_main.imshow(cur_img)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    label = "blur" if stage == 0 else "downsample"
    ax_main.set_title(label, fontsize=11, color=ACTIVE_COLOR)
    level_shown = step_idx if stage == 0 else step_idx + 1
    ax_main.set_xlabel(f"level {level_shown}: {cur_img.shape[1]}x{cur_img.shape[0]}",
                         fontsize=10, color=NEUTRAL_COLOR)

    draw_pancake_stack(ax_stack, done_levels if not final_hold else NUM_LEVELS)
    ax_stack.set_title("pyramid", fontsize=10)

    draw_zoomed_stack(ax_zoom, done_levels if not final_hold else NUM_LEVELS)
    ax_zoom.set_title("resized", fontsize=10)

    fig.suptitle("building the pyramid", fontsize=12.5,
                   color=NEUTRAL_COLOR, fontweight="bold")
    if final_hold:
        fig.text(0.5, 0.895, f"{NUM_LEVELS} levels, each half the size and visibly blurrier than the last",
                   ha="center", va="top", fontsize=11, color=MATCH_COLOR, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STAGE = 10
HOLD_LEVEL_END = 16
HOLD_END = 44

for step_idx in range(NUM_LEVELS - 1):
    frames += [render(0, step_idx)] * HOLD_STAGE
    frames += [render(1, step_idx)] * HOLD_LEVEL_END

frames += [render(1, NUM_LEVELS - 2, final_hold=True)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=90,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
