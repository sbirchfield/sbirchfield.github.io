"""Generate lens_distortion_demo.gif: an animated version of the lesson 27
notebook's "Straight lines, bent by a lens" / "Undoing distortion" cells.
Reuses that cell's exact grid-line construction, K_true, dist_true,
K_estimated, and dist_estimated.

Two phases:
  - distort: k1, k2 (and the rest of dist_true, scaled the same way) ramp
    from 0 up to their true values, bowing the straight grid lines into the
    barrel-distorted shape seen "through the lens".
  - undistort: the bent grid is pulled back toward straight using the
    *estimated* calibration (not the true one), matching the notebook's own
    undistorted_px result -- nearly straight, with a faint residual bow.

Run from anywhere:
    python make_lens_distortion_demo.py
Output:
    lens_distortion_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "lens_distortion_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"

# Same synthetic rig as the notebook.
K_true = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
dist_true = np.array([-0.3, 0.1, 0.001, -0.0005, 0.02])  # k1, k2, p1, p2, k3

rng = np.random.default_rng(0)
cols, rows = 9, 6
corners_3d = np.zeros((rows * cols, 3), dtype=np.float64)
corners_3d[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
corners_3d[:, :2] -= corners_3d[:, :2].mean(axis=0)

object_points, image_points = [], []
for _ in range(40):
    rvec = rng.uniform(-0.4, 0.4, 3)
    tvec = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4), rng.uniform(9, 13)])
    projected, _ = cv2.projectPoints(corners_3d, rvec, tvec, K_true, dist_true)
    projected = projected.reshape(-1, 2)
    if np.all(projected >= 5) and np.all(projected[:, 0] < 640 - 5) and np.all(projected[:, 1] < 480 - 5):
        object_points.append(corners_3d.astype(np.float32))
        image_points.append(projected.astype(np.float32))

detected_object_points, detected_image_points = [], []
for obj_pts, img_pts in zip(object_points, image_points):
    detected_object_points.append(obj_pts)
    detected_image_points.append(img_pts)

_, K_estimated, dist_estimated, _, _ = cv2.calibrateCamera(
    detected_object_points, detected_image_points, (640, 480), None, None)


def distort_normalized(xy, dist):
    k1, k2, p1, p2, k3 = dist
    x, y = xy[:, 0], xy[:, 1]
    r2 = x**2 + y**2
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
    yd = y * radial + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
    return np.stack([xd, yd], axis=1)


def make_grid_lines():
    lines = []
    for xv in np.linspace(-0.38, 0.38, 9):
        ys = np.linspace(-0.28, 0.28, 50)
        lines.append(np.stack([np.full_like(ys, xv), ys], axis=1))
    for yv in np.linspace(-0.28, 0.28, 7):
        xs = np.linspace(-0.38, 0.38, 50)
        lines.append(np.stack([xs, np.full_like(xs, yv)], axis=1))
    return lines


grid_lines_normalized = make_grid_lines()
straight_px = [line @ K_true[:2, :2].T + K_true[:2, 2] for line in grid_lines_normalized]
distorted_px = [distort_normalized(line, dist_true) @ K_true[:2, :2].T + K_true[:2, 2]
                for line in grid_lines_normalized]
undistorted_normalized = [
    cv2.undistortPoints(px.reshape(-1, 1, 2).astype(np.float64), K_estimated, dist_estimated).reshape(-1, 2)
    for px in distorted_px
]
undistorted_px = [line @ K_estimated[:2, :2].T + K_estimated[:2, 2] for line in undistorted_normalized]


def lines_at_dist_scale(t):
    """Grid lines distorted by dist_true * t, in pixel coordinates."""
    scaled = dist_true * t
    return [distort_normalized(line, scaled) @ K_true[:2, :2].T + K_true[:2, 2]
            for line in grid_lines_normalized]


def lines_undistort_blend(t):
    """Linear pixel-space blend from fully distorted (t=0) to undistorted (t=1) --
    a visual proxy for the undistortion step, not a re-run of the model."""
    return [d + t * (u - d) for d, u in zip(distorted_px, undistorted_px)]


def draw_lines(ax, lines_px, color, height=480, width=640):
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for line in lines_px:
        ax.plot(line[:, 0], line[:, 1], color=color, linewidth=1.1)


def render(lines_px, title, color):
    fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=120)
    fig.subplots_adjust(top=0.88, bottom=0.03, left=0.03, right=0.97)
    draw_lines(ax, lines_px, NEUTRAL_COLOR)
    fig.suptitle(title, fontsize=12.5, color=color, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_START = 14
HOLD_STEP = 3
HOLD_MID = 20
HOLD_END = 44

N_DISTORT = 24
for t in np.linspace(0, 1, N_DISTORT):
    k1 = dist_true[0] * t
    k2 = dist_true[1] * t
    title = f"distorting:  k1={k1:+.3f}  k2={k2:+.3f}"
    frames += [render(lines_at_dist_scale(t), title, ACTIVE_COLOR)] * (HOLD_START if t == 0 else HOLD_STEP)

frames += [render(distorted_px, "through the lens (true k1, k2)", ACTIVE_COLOR)] * HOLD_MID

N_UNDISTORT = 24
for t in np.linspace(0, 1, N_UNDISTORT):
    title = "undistorting using estimated K, dist"
    frames += [render(lines_undistort_blend(t), title, MATCH_COLOR)] * HOLD_STEP

frames += [render(undistorted_px, "undistorted (faint residual bow remains)", MATCH_COLOR)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=70,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
