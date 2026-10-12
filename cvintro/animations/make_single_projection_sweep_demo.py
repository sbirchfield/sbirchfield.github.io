"""Generate single_projection_sweep_demo.gif: an animated companion to the
lesson 30 notebook's "What a single projection cannot do" cell. Sweeps a
single projection direction w through every angle (0 to pi; w and -w give
the same decision line), picking the best threshold b at each angle, on the
notebook's own ring-vs-disk dataset -- showing live that no rotation or
shift of one straight line ever comes close to separating the ring from the
disk it surrounds.

Reuses the notebook cell's own rng seed/data generation and
plot_projection_classifier() layout (2D boundary + 1D projected scores) so
the animation is visually consistent with the static figure it accompanies.

Run from anywhere:
    python make_single_projection_sweep_demo.py
Output:
    single_projection_sweep_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "single_projection_sweep_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"

# Same rng draws (in the same order) as the notebook, so inner/outer match exactly.
rng = np.random.default_rng(1)
_ = rng.multivariate_normal([0, 0], [[3, 1.5], [1.5, 1]], 200)       # cloud (unused here)
_ = rng.normal(size=10)                                               # signal (unused here)
_ = rng.normal(loc=[-2, -1], scale=0.8, size=(60, 2))                 # class_a (unused here)
_ = rng.normal(loc=[2, 1.5], scale=0.8, size=(60, 2))                 # class_b (unused here)

theta_in = rng.uniform(0, 2 * np.pi, 60)
inner = np.stack([0.5 * np.cos(theta_in), 0.5 * np.sin(theta_in)], axis=1) + rng.normal(0, 0.1, (60, 2))
theta_out = rng.uniform(0, 2 * np.pi, 60)
outer = np.stack([2.0 * np.cos(theta_out), 2.0 * np.sin(theta_out)], axis=1) + rng.normal(0, 0.15, (60, 2))

B_RANGE = np.linspace(-3, 3, 61)


def best_threshold(w):
    s_in, s_out = inner @ w, outer @ w
    best_acc, best_b = 0.0, 0.0
    for b in B_RANGE:
        acc = max((s_in + b < 0).sum() + (s_out + b > 0).sum(),
                  (s_in + b > 0).sum() + (s_out + b < 0).sum()) / 120
        if acc > best_acc:
            best_acc, best_b = acc, b
    return best_b, best_acc


def render(angle, best_so_far):
    w = np.array([np.cos(angle), np.sin(angle)])
    b, acc = best_threshold(w)
    score_in, score_out = inner @ w + b, outer @ w + b

    best_so_far = max(best_so_far, acc)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), dpi=120)

    ax = axes[0]
    ax.scatter(*inner.T, s=15, label="inner (disk)")
    ax.scatter(*outer.T, s=15, label="outer (ring)")
    foot = -b * w
    perp = np.array([-w[1], w[0]])
    p1, p2 = foot + 5 * perp, foot - 5 * perp
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black", linewidth=1.5, label="decision boundary")
    ax.arrow(*foot, *w, head_width=0.15, color=ACTIVE_COLOR, length_includes_head=True)
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
    ax.set_aspect("equal")
    ax.set_title(f"w at {np.degrees(angle):5.0f}°, best threshold  (accuracy = {acc:.0%})", fontsize=9.5)
    ax.legend(fontsize=7, loc="lower right")

    ax2 = axes[1]
    ax2.scatter(score_in, np.full_like(score_in, 0.013), s=15, label="inner (disk)")
    ax2.scatter(score_out, np.zeros_like(score_out), s=15, label="outer (ring)")
    ax2.axvline(0, color="black", linewidth=1.5, label="threshold")
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-0.2, 0.2)
    ax2.set_yticks([])
    ax2.set_title("the same points, projected to 1D", fontsize=9.5)
    ax2.legend(fontsize=7)

    color = MATCH_COLOR if best_so_far >= 0.85 else NEUTRAL_COLOR
    fig.suptitle(f"no single straight line separates the ring from the disk  "
                 f"(best found so far: {best_so_far:.0%})",
                 fontsize=11.5, color=color, fontweight="bold")
    fig.subplots_adjust(top=0.84, bottom=0.08, wspace=0.25)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB"), best_so_far


frames = []
HOLD_STEP = 2
HOLD_END = 40

angles = np.linspace(0, np.pi, 72, endpoint=False)
best_so_far, best_angle = 0.0, angles[0]
for a in angles:
    frame, best_so_far_new = render(a, best_so_far)
    if best_so_far_new > best_so_far:
        best_angle = a
    best_so_far = best_so_far_new
    frames += [frame] * HOLD_STEP

final_frame, best_so_far = render(best_angle, best_so_far)
frames += [final_frame] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=60,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames, best accuracy found: {best_so_far:.1%})")
