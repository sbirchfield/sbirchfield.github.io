"""Generate transform_hierarchy_demo.gif: an animated version of the lesson 8
notebook's "Seeing the difference on a unit square" cell. Instead of showing
only the before/after end states of a Euclidean, similarity, and affine
transform, this morphs a unit square continuously from identity to each
final transform (and back), so it's visually obvious which properties
(lengths, angles, parallelism) survive the trip and which don't.

Uses the exact same parameters as the notebook cell (theta=30 deg,
t=(0.3, 0.1), similarity scale 1.6, affine matrix [[1.6, 0.7], [0.2, 0.9]]).

Run from anywhere:
    python make_transform_hierarchy_demo.py
Output:
    transform_hierarchy_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "transform_hierarchy_demo.gif")

square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.float64)

THETA = np.radians(30)
T_FINAL = np.array([0.3, 0.1])
SIM_SCALE = 1.6
AFFINE_A = np.array([[1.6, 0.7], [0.2, 0.9]])

ACTIVE_COLOR = "#e74c3c"
ORIG_COLOR = "#999999"


def rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])


def apply_2x2(pts, A, t):
    return pts @ A.T + t


def panel_at(alpha, kind):
    t = T_FINAL * alpha
    if kind == "euclidean":
        A = rot(THETA * alpha)
    elif kind == "similarity":
        s = 1 + (SIM_SCALE - 1) * alpha
        A = s * rot(THETA * alpha)
    else:  # affine
        A = np.eye(2) + alpha * (AFFINE_A - np.eye(2))
    return apply_2x2(square, A, t)


PANELS = [
    ("euclidean", "Euclidean", "preserves lengths & angles"),
    ("similarity", "Similarity", "preserves angles, ratios of lengths"),
    ("affine", "Affine", "preserves parallelism only"),
]


def render(alpha):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=120)
    fig.subplots_adjust(top=0.92, bottom=0.04, wspace=0.3)
    for ax, (kind, title, caption) in zip(axes, PANELS):
        pts = panel_at(alpha, kind)
        ax.plot(*square.T, '--', color=ORIG_COLOR, linewidth=1.6, label="original")
        ax.plot(*pts.T, color=ACTIVE_COLOR, linewidth=2.8, label="transformed")
        ax.set_aspect("equal")
        ax.set_xlim(-0.6, 2.7)
        ax.set_ylim(-0.15, 2.4)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.text(0.5, -0.08, caption, transform=ax.transAxes, ha="center", va="top",
                 fontsize=11, color="#555555")
        ax.legend(fontsize=9.5, loc="upper left")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


N = 30  # steps from alpha=0 to alpha=1
HOLD_END = 10

alphas_fwd = np.linspace(0.0, 1.0, N + 1)
alphas_bwd = alphas_fwd[::-1][1:-1]

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
