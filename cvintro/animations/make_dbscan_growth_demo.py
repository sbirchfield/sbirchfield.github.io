"""Generate dbscan_growth_demo.gif: an animated walkthrough of DBSCAN's
region-growing process (the `dbscan` function in the lesson 19 notebook),
run on the same two-concentric-rings dataset used there.

Snapshots the algorithm's state (visited, core, border, noise, frontier)
every time a batch of new points get visited, in the same "growing frontier"
style as make_floodfill_demo.py -- except this frontier can jump across the
ring rather than only flood through 4-connected neighbors, since DBSCAN's
neighborhoods are eps-balls in point space, not grid adjacency.

Run from anywhere:
    python make_dbscan_growth_demo.py
Output:
    dbscan_growth_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "dbscan_growth_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"
NOISE_COLOR = "#bbbbbb"
UNVISITED_COLOR = "#dddddd"
CLUSTER_COLORS = ["#4c72b0", "#c44e52"]

EPS = 0.6
MIN_SAMPLES = 4

# Same dataset construction as the lesson 19 notebook (seed=1, ring(150, 1.0)
# and ring(150, 2.5)).
rng = np.random.default_rng(1)


def ring(n, r, noise=0.05):
    theta = rng.uniform(0, 2 * np.pi, n)
    rad = r + rng.normal(0, noise, n)
    return np.stack([rad * np.cos(theta), rad * np.sin(theta)], axis=1)


X = np.vstack([ring(150, 1.0), ring(150, 2.5)])
n = len(X)
dist = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))


def dbscan_steps(X, eps, min_samples):
    """Same algorithm as the notebook's dbscan(), but snapshotting
    (labels, visited, frontier) every time new points get visited."""
    labels = np.full(n, -1)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0
    steps = [(labels.copy(), visited.copy(), set())]
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = list(np.where(dist[i] <= eps)[0])
        if len(neighbors) < min_samples:
            steps.append((labels.copy(), visited.copy(), {i}))
            continue
        labels[i] = cluster_id
        j = 0
        while j < len(neighbors):
            q = neighbors[j]
            if not visited[q]:
                visited[q] = True
                q_neighbors = np.where(dist[q] <= eps)[0]
                if len(q_neighbors) >= min_samples:
                    neighbors.extend([x for x in q_neighbors if x not in neighbors])
            if labels[q] == -1:
                labels[q] = cluster_id
            j += 1
            if j % 6 == 0:
                steps.append((labels.copy(), visited.copy(), set(neighbors[j:j + 6])))
        steps.append((labels.copy(), visited.copy(), set(neighbors[j:])))
        cluster_id += 1
    steps.append((labels.copy(), visited.copy(), set()))
    return steps


steps = dbscan_steps(X, EPS, MIN_SAMPLES)


def render(step_i, final_hold=False):
    labels, visited, frontier = steps[step_i]

    fig, ax = plt.subplots(figsize=(6.6, 6.0), dpi=120)
    fig.subplots_adjust(top=0.94, bottom=0.04)
    ax.set_aspect("equal")
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-3.3, 3.3)

    colors = []
    for i in range(n):
        if not visited[i]:
            colors.append(UNVISITED_COLOR)
        elif labels[i] == -1:
            colors.append(NOISE_COLOR)
        else:
            colors.append(CLUSTER_COLORS[labels[i] % len(CLUSTER_COLORS)])
    ax.scatter(*X.T, c=colors, s=16, zorder=2, edgecolors="none")

    if frontier:
        idx = np.array(sorted(frontier))
        # faint eps-radius circle around each frontier point -- makes the
        # "neighborhood" DBSCAN is actually querying directly visible,
        # instead of just asserting the jump is eps-based in prose
        for x0, y0 in X[idx]:
            ax.add_patch(plt.Circle((x0, y0), EPS, facecolor="none",
                                      edgecolor="#999999", linewidth=0.8, alpha=0.6, zorder=1))
        ax.scatter(*X[idx].T, facecolors="none", edgecolors=ACTIVE_COLOR,
                   s=70, linewidths=1.8, zorder=3)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    n_visited = int(visited.sum())
    n_clusters = len(set(labels[labels >= 0].tolist()))
    n_noise = int(((labels == -1) & visited).sum())
    if final_hold:
        title = f"done: {n_clusters} clusters found, {n_noise} points labeled noise"
        color = MATCH_COLOR
    else:
        title = f"visited {n_visited}/{n}  |  clusters so far: {n_clusters}  |  noise: {n_noise}"
        color = ACTIVE_COLOR
    fig.suptitle(title, fontsize=12, color=color, fontweight="bold" if final_hold else "normal")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 2
HOLD_END = 44

for i in range(len(steps)):
    frames += [render(i)] * HOLD_STEP
frames += [render(len(steps) - 1, final_hold=True)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=70,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames, {len(steps)} steps)")
