"""Generate lucas_kanade_tracking_demo.gif: a short synthetic "video" made by
warping a single real photo (building.png, already used in lesson 20) with a
smoothly changing affine transform per frame -- simulating a panning,
slightly zooming camera. Shi-Tomasi corners are detected on frame 0, then
tracked frame-to-frame with the real cv2.calcOpticalFlowPyrLK (the notebook's
own Lucas-Kanade, run iteratively/pyramidally exactly as OpenCV implements
it), drawing each point's accumulated trajectory as the sequence plays.

Run from anywhere:
    python make_lucas_kanade_tracking_demo.py
Output:
    lucas_kanade_tracking_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(OUT_DIR, "..", "img", "building.png")
OUT_GIF = os.path.join(OUT_DIR, "lucas_kanade_tracking_demo.gif")

NEUTRAL_COLOR = "#222222"

base = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
h, w = base.shape
center = (w / 2, h / 2)

N_FRAMES = 14
frames_gray = []
for t in range(N_FRAMES):
    angle = 1.1 * t
    scale = 1.0 - 0.006 * t
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += 1.8 * t
    M[1, 2] += 1.0 * t
    frames_gray.append(cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE))

# Detect once on frame 0 (same Shi-Tomasi "good features to track" as Lesson 20),
# then chain cv2.calcOpticalFlowPyrLK frame-to-frame -- real pyramidal
# Lucas-Kanade, the same function the notebook's "larger motions" cell uses.
pts0 = cv2.goodFeaturesToTrack(frames_gray[0], maxCorners=25, qualityLevel=0.1, minDistance=12)
n_pts = len(pts0)

LK_PARAMS = dict(winSize=(15, 15), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

trajectories = [[pts0[i, 0].copy()] for i in range(n_pts)]
alive = np.ones(n_pts, dtype=bool)
prev_pts = pts0.copy()
prev_gray = frames_gray[0]
for t in range(1, N_FRAMES):
    next_gray = frames_gray[t]
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_pts, None, **LK_PARAMS)
    status = status.reshape(-1)
    for i in range(n_pts):
        if alive[i]:
            if status[i] == 1:
                trajectories[i].append(next_pts[i, 0].copy())
            else:
                alive[i] = False
    prev_pts = next_pts
    prev_gray = next_gray

TRACK_COLOR = "#e02020"


def render(t):
    vis = cv2.cvtColor(frames_gray[t], cv2.COLOR_GRAY2RGB)
    fig, ax = plt.subplots(figsize=(4.6, 6.4), dpi=120)
    fig.subplots_adjust(top=0.90, bottom=0.03, left=0.03, right=0.97)
    ax.imshow(vis)
    for traj in trajectories:
        shown = traj[:min(t + 1, len(traj))]
        xs = [p[0] for p in shown]
        ys = [p[1] for p in shown]
        # all one color, thick enough to read clearly against the photo --
        # per-point colors from a 20-color colormap looked nice but several
        # were too close to the background or to each other to tell apart
        ax.plot(xs, ys, color=TRACK_COLOR, linewidth=2.2, alpha=0.9, zorder=2)
        ax.scatter([xs[-1]], [ys[-1]], color=TRACK_COLOR, s=32, zorder=3,
                    edgecolors="white", linewidths=0.8)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    n_alive = sum(1 for traj in trajectories if len(traj) > t)
    fig.suptitle(f"Lucas-Kanade tracking: frame {t + 1}/{N_FRAMES}, {n_alive}/{n_pts} points tracked",
                 fontsize=11, color=NEUTRAL_COLOR)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_FRAME = 10
HOLD_END = 36

for t in range(N_FRAMES):
    frames += [render(t)] * HOLD_FRAME
frames += [render(N_FRAMES - 1)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=90,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames, {n_pts} points, "
      f"{sum(1 for t in trajectories if len(t) == N_FRAMES)} survived to the end)")
