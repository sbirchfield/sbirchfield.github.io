"""Generate ransac_demo.gif: an animated version of the lesson 23 notebook's
`ransac_line` function. Reuses that cell's exact outlier-contaminated data
(rng seed 0, 40 inliers + 15 outliers) and exact algorithm (sample 2 points,
fit a line, count inliers within `threshold`, keep the best-so-far), run
with the notebook's own rng seed 1 so the animation's final result matches
the static cell exactly.

Each iteration: highlight the 2 randomly sampled points, draw the candidate
line through them, color every point inlier/outlier under that candidate,
and show the running best model updating whenever a new candidate beats it.
Ends with the final least-squares refit on the winning inlier set.

Run from anywhere:
    python make_ransac_demo.py
Output:
    ransac_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "ransac_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"
INLIER_COLOR = "#4c72b0"
OUTLIER_COLOR = "#c44e52"

# Same data as the notebook's "Why least squares breaks under outliers" cell.
rng = np.random.default_rng(0)
true_slope, true_intercept = 2.0, 5.0

x_inliers = rng.uniform(0, 10, 40)
y_inliers = true_slope * x_inliers + true_intercept + rng.normal(0, 0.5, 40)

x_outliers = rng.uniform(0, 10, 15)
y_outliers = rng.uniform(-20, 40, 15)

x = np.concatenate([x_inliers, x_outliers])
y = np.concatenate([y_inliers, y_outliers])
XS = np.array([0.0, 10.0])
THRESHOLD = 2.0
N_ITERATIONS = 200


def line_through(i, j):
    if x[i] == x[j]:
        return None
    m = (y[j] - y[i]) / (x[j] - x[i])
    b = y[i] - m * x[i]
    return m, b


def inliers_for(m, b):
    distance = np.abs(m * x - y + b) / np.sqrt(m**2 + 1)
    return distance < THRESHOLD


# Precompute the exact same trial sequence as ransac_line(..., rng=seed 1),
# snapshotting every trial plus whenever a new best is found.
trial_rng = np.random.default_rng(1)
best_inliers, best_count = None, -1
best_mb = None
trials = []  # (i, j, m, b, inliers, is_new_best, best_count_so_far, best_mb_so_far)
for _ in range(N_ITERATIONS):
    i, j = trial_rng.choice(len(x), 2, replace=False)
    line = line_through(i, j)
    if line is None:
        continue
    m, b = line
    inliers = inliers_for(m, b)
    is_new_best = inliers.sum() > best_count
    if is_new_best:
        best_count, best_inliers = inliers.sum(), inliers
        best_mb = (m, b)
    trials.append((i, j, m, b, inliers, is_new_best, best_count, best_mb))

A = np.vstack([x[best_inliers], np.ones(best_inliers.sum())]).T
final_slope, final_intercept = np.linalg.lstsq(A, y[best_inliers], rcond=None)[0]


def render(trial_idx, final=False):
    fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=120)
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.08, right=0.97)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-22, 45)

    if final:
        i, j, m, b, cand_inliers, is_new_best, count, mb = trials[-1]
        inliers_mask = best_inliers
        title1 = "final refit on the winning inlier set"
        title2 = f"RANSAC fit:  y = {final_slope:4.2f}x + {final_intercept:4.2f}  ({best_count:3d}/{len(x)} inliers)"
        color = MATCH_COLOR
    else:
        i, j, m, b, cand_inliers, is_new_best, count, mb = trials[trial_idx]
        inliers_mask = cand_inliers
        title1 = f"trial {trial_idx + 1:3d}/{N_ITERATIONS}"
        title2 = f"candidate: {int(cand_inliers.sum()):3d} inliers   |   best so far: {count:3d} inliers"
        color = ACTIVE_COLOR if not is_new_best else MATCH_COLOR

    ax.scatter(x[~inliers_mask], y[~inliers_mask], c=OUTLIER_COLOR, marker="x",
               s=28, zorder=2, label="outside threshold")
    ax.scatter(x[inliers_mask], y[inliers_mask], c=INLIER_COLOR, s=28,
               zorder=2, label="within threshold")

    if not final:
        ax.scatter(x[[i, j]], y[[i, j]], facecolors="none", edgecolors=ACTIVE_COLOR,
                   s=160, linewidths=2.0, zorder=4)
        ax.plot(XS, m * XS + b, color=ACTIVE_COLOR, linewidth=2.0, zorder=3,
                label="candidate line")

    best_m, best_b = mb if not final else (final_slope, final_intercept)
    ax.plot(XS, best_m * XS + best_b, color=MATCH_COLOR, linewidth=2.2,
            zorder=3, linestyle="--" if not final else "-",
            label="best-so-far" if not final else "RANSAC fit")

    ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.suptitle(title1, fontsize=12.5, color=color, fontweight="bold")
    fig.text(0.5, 0.895, title2, ha="center", va="top", fontsize=10.5,
              color=NEUTRAL_COLOR, family="monospace")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_TRIAL = 2
HOLD_BEST = 14
HOLD_END = 44

for t in range(len(trials)):
    is_new_best = trials[t][5]
    frames += [render(t)] * (HOLD_BEST if is_new_best else HOLD_TRIAL)

frames += [render(None, final=True)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=60,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames, {len(trials)} trials, {best_count} final inliers)")
