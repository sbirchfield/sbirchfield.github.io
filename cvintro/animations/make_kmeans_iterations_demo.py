"""Generate kmeans_iterations_demo.gif: an animated version of Lloyd's
algorithm (k-means), illustrating the lesson 19 notebook's `kmeans` function.
Uses a small synthetic 2D point cloud with 3 well-separated round blobs (not
the notebook's own L*a*b* pixel data, which isn't 2D-plottable) so the
assign/recompute cycle is directly visible on a scatter plot.

Alternates two kinds of steps each iteration:
  - "assign": every point recolors to its nearest current center
  - "update": center markers jump to the mean of their assigned points
until centers stop moving (convergence), then holds on a "converged" frame.

Run from anywhere:
    python make_kmeans_iterations_demo.py
Output:
    kmeans_iterations_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "kmeans_iterations_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"
CLUSTER_COLORS = ["#4c72b0", "#dd8452", "#55a868"]
K = 3

rng = np.random.default_rng(2)
true_centers = np.array([[-3.0, 2.0], [3.0, 2.0], [0.0, -3.0]])
X = np.vstack([c + rng.normal(0, 0.9, (40, 2)) for c in true_centers])

# Deliberately poor initial centers (clustered together) so several
# iterations are needed before convergence, making the animation worth watching.
init_centers = np.array([[-1.0, 3.5], [0.5, 3.5], [1.5, 3.0]])


def assign(X, centers):
    dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    return dists.argmin(1)


def update(X, labels, centers, k):
    return np.array([X[labels == j].mean(0) if (labels == j).any() else centers[j]
                      for j in range(k)])


# Precompute the sequence of (centers_before, labels, centers_after) steps.
steps = []
centers = init_centers.copy()
for it in range(8):
    labels = assign(X, centers)
    new_centers = update(X, labels, centers, K)
    steps.append((centers.copy(), labels, new_centers.copy()))
    if np.allclose(new_centers, centers):
        centers = new_centers
        break
    centers = new_centers
final_centers = centers


GRAY = "#999999"


def render(step_i, phase, converged=False):
    """phase: 'init' (nothing assigned yet), 'assign' (points recolor to
    nearest of centers_before), or 'update' (centers jump from
    centers_before to centers_after)."""
    centers_before, labels, centers_after = steps[step_i]

    fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=120)
    fig.subplots_adjust(top=0.74, bottom=0.02)
    ax.set_aspect("equal")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)

    if phase == "init":
        ax.scatter(*X.T, c=GRAY, s=28, zorder=2, edgecolors="white", linewidths=0.4)
        ax.scatter(*centers_before.T, c=CLUSTER_COLORS, s=240, marker="X",
                    zorder=4, edgecolors="black", linewidths=1.5)
    else:
        colors = [CLUSTER_COLORS[l] for l in labels]
        ax.scatter(*X.T, c=colors, s=28, zorder=2, edgecolors="white", linewidths=0.4)

        if phase == "assign":
            # centers are colored per-cluster (not a single generic color)
            # so it's visually obvious which center each point was just
            # assigned to
            ax.scatter(*centers_before.T, c=CLUSTER_COLORS, s=240, marker="X",
                        zorder=4, edgecolors="black", linewidths=1.5)
        else:
            ax.scatter(*centers_before.T, c=CLUSTER_COLORS, s=130, marker="X",
                        zorder=3, edgecolors="black", linewidths=0.8, alpha=0.45)
            for (c0, c1, col) in zip(centers_before, centers_after, CLUSTER_COLORS):
                ax.annotate("", xy=c1, xytext=c0,
                            arrowprops=dict(arrowstyle="->", color=col, lw=1.8))
            ax.scatter(*centers_after.T, c=CLUSTER_COLORS, s=240, marker="X",
                        zorder=4, edgecolors="black", linewidths=1.5)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    title_color = MATCH_COLOR if converged else ACTIVE_COLOR
    if phase == "init":
        fig.text(0.5, 0.98, "initial centers (no points assigned yet)",
                  ha="center", va="top", fontsize=12.5, color=NEUTRAL_COLOR)
    elif converged:
        fig.text(0.5, 0.98, f"iteration {step_i + 1}: converged, centers no longer move",
                  ha="center", va="top", fontsize=12.5, color=title_color, fontweight="bold")
    else:
        # "assign"/"update" gets its own bold line, separate from the
        # (smaller) description of what that step does -- easier to read
        # at a glance than cramming the word and its explanation together
        word = "ASSIGN" if phase == "assign" else "UPDATE"
        desc = "each point -> nearest center" if phase == "assign" else "center -> mean of its assigned points"
        fig.text(0.5, 0.98, f"iteration {step_i + 1}", ha="center", va="top",
                  fontsize=11, color=NEUTRAL_COLOR)
        fig.text(0.5, 0.90, word, ha="center", va="top",
                  fontsize=14, color=title_color, fontweight="bold")
        fig.text(0.5, 0.825, desc, ha="center", va="top", fontsize=10.5, color=title_color)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_INIT = 16
HOLD_ASSIGN = 10
HOLD_UPDATE = 14
HOLD_END = 44

frames += [render(0, "init")] * HOLD_INIT

for i in range(len(steps)):
    frames += [render(i, "assign")] * HOLD_ASSIGN
    frames += [render(i, "update")] * HOLD_UPDATE

frames += [render(len(steps) - 1, "update", converged=True)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=90,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames, {len(steps)} iterations)")
