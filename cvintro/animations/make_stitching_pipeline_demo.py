"""Generate stitching_pipeline_demo.gif: an animated montage of the lesson 25
notebook's full stitching pipeline, run on the same two clemson photos used
there -- match, robust homography, warp, and feathered blend -- shown as one
continuous sequence building the panorama, instead of four static before/
after panels.

Stages:
  1. the two input photos side by side
  2. SIFT + ratio-test matches drawn between them (same good_matches as the
     notebook)
  3. the RANSAC-fit homography's warped canvas appearing (view 2 warped into
     view 1's frame)
  4. the feathered blend completing, overlap region fading in via the same
     linear alpha ramp as the notebook

Run from anywhere:
    python make_stitching_pipeline_demo.py
Output:
    stitching_pipeline_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "stitching_pipeline_demo.gif")
IMG_DIR = os.path.join(OUT_DIR, "..", "img")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"

view1 = cv2.imread(os.path.join(IMG_DIR, "clemson00.jpg"))
view2 = cv2.imread(os.path.join(IMG_DIR, "clemson01.jpg"))

gray1 = cv2.cvtColor(view1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(view2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher()
raw_matches = bf.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]

pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
H, inlier_mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0)
mask = inlier_mask.ravel().astype(bool)

h1, w1 = view1.shape[:2]
h2, w2 = view2.shape[:2]
corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
warped_corners2 = cv2.perspectiveTransform(corners2, H)
all_corners = np.concatenate([corners1, warped_corners2], axis=0)

x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)
canvas_w, canvas_h = x_max - x_min, y_max - y_min

translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
canvas1 = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
canvas1[-y_min:-y_min + h1, -x_min:-x_min + w1] = view1
warped2 = cv2.warpPerspective(view2, translation @ H, (canvas_w, canvas_h))

has1 = canvas1.sum(axis=2) > 0
has2 = warped2.sum(axis=2) > 0
overlap = has1 & has2

overlap_cols = np.where(overlap.any(axis=0))[0]
x_start, x_end = overlap_cols.min(), overlap_cols.max()
ramp = np.clip((np.arange(canvas_w) - x_start) / (x_end - x_start + 1e-6), 0, 1)

alpha_full = np.zeros((canvas_h, canvas_w), dtype=np.float32)
alpha_full[overlap] = np.broadcast_to(ramp, (canvas_h, canvas_w))[overlap]


def blend_at(frac):
    """frac in [0, 1]: 0 = canvas1 alone with warped2 not yet composited,
    1 = the final feathered blend. Scales the alpha ramp's magnitude so the
    overlap fades in smoothly rather than popping in at full blend."""
    alpha = alpha_full * frac
    out = canvas1.astype(np.float32) * (1 - alpha[..., None]) + warped2.astype(np.float32) * alpha[..., None]
    out[has1 & ~has2] = canvas1[has1 & ~has2]
    out[has2 & ~has1] = warped2[has2 & ~has1] * frac  # view-2-only region also fades in
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def render_side_by_side(title, color):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), dpi=120)
    fig.subplots_adjust(top=0.84, bottom=0.04, wspace=0.05)
    axes[0].imshow(to_rgb(view1))
    axes[0].set_title("view 1", fontsize=10)
    axes[1].imshow(to_rgb(view2))
    axes[1].set_title("view 2", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=12, color=color, fontweight="bold")
    return fig


def render_matches(n_shown, title, color):
    match_vis = cv2.drawMatches(view1, kp1, view2, kp2, good_matches[:n_shown], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=120)
    fig.subplots_adjust(top=0.82, bottom=0.02, left=0.01, right=0.99)
    ax.imshow(to_rgb(match_vis))
    ax.axis("off")
    fig.suptitle(title, fontsize=12, color=color, fontweight="bold")
    return fig


def render_canvas(img, title, color):
    fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=120)
    fig.subplots_adjust(top=0.82, bottom=0.02, left=0.01, right=0.99)
    ax.imshow(to_rgb(img))
    ax.axis("off")
    fig.suptitle(title, fontsize=12, color=color, fontweight="bold")
    return fig


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_INPUT = 16
HOLD_MATCH_STEP = 2
HOLD_MATCH_END = 14
HOLD_WARP = 16
N_BLEND = 20
HOLD_BLEND_STEP = 2
HOLD_END = 44

# 1. the two input photos
frames += [fig_to_image(render_side_by_side("two overlapping photos", NEUTRAL_COLOR))] * HOLD_INPUT

# 2. matches appearing a batch at a time
match_counts = list(range(4, min(60, len(good_matches)) + 1, 4))
for n in match_counts:
    frames += [fig_to_image(render_matches(
        n, f"step 1: SIFT + ratio test -- {n} good matches shown", ACTIVE_COLOR))] * HOLD_MATCH_STEP
frames += [fig_to_image(render_matches(
    min(60, len(good_matches)),
    f"step 1: {len(good_matches)} good matches found ({int(mask.sum())} RANSAC inliers)",
    MATCH_COLOR))] * HOLD_MATCH_END

# 3. the warped canvas (view 2 alone, pre-blend) appearing next to view 1's placement
canvas_view2_only = np.zeros_like(canvas1)
canvas_view2_only[has2] = warped2[has2]
frames += [fig_to_image(render_canvas(
    canvas_view2_only, "step 2-3: robust homography (RANSAC) warps view 2 into view 1's frame",
    ACTIVE_COLOR))] * HOLD_WARP

# 4. feathered blend fading in
for i in range(N_BLEND + 1):
    frac = i / N_BLEND
    frames += [fig_to_image(render_canvas(
        blend_at(frac), "step 4: feathered blend across the overlap", ACTIVE_COLOR))] * HOLD_BLEND_STEP

frames += [fig_to_image(render_canvas(
    blend_at(1.0), "stitched panorama", MATCH_COLOR))] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=80,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
