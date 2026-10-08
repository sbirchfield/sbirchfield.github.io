"""Generate eigen_demo.gif: an animated walkthrough of what a matrix's
eigenvectors mean geometrically -- a unit circle of vectors is morphed
through the matrix A, and the eigenvectors are the only directions that
don't rotate during the whole process, they only scale.

Run from anywhere:
    python make_eigen_demo.py
Output:
    eigen_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "eigen_demo.gif")

# Same matrix used in the lesson 6 notebook's eigenvector refresher cell.
A = np.array([[3.0, 1.0],
              [1.0, 1.5]])

eigvals, eigvecs = np.linalg.eigh(A)  # ascending order
order = [1, 0]  # index 0 = major (largest eigenvalue), index 1 = minor
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

MAJOR_COLOR = "#e54040"  # red: eigenvector with the larger eigenvalue
MINOR_COLOR = "#1f6fd6"  # blue: eigenvector with the smaller eigenvalue
VEC_COLOR = "#999999"    # gray: ordinary (non-eigen) vectors

N_ARROWS = 12
arrow_angles = np.linspace(0, 2 * np.pi, N_ARROWS, endpoint=False)
arrow_starts = np.stack([np.cos(arrow_angles), np.sin(arrow_angles)])  # 2xN

THETA = np.linspace(0, 2 * np.pi, 200)
CIRCLE = np.stack([np.cos(THETA), np.sin(THETA)])  # 2x200

LIMIT = eigvals[0] * 1.25  # largest eigenvalue sets the final ellipse extent


def draw_matrix_panel(ax):
    """Static panel shown every frame: the matrix A itself, spelled out so
    there is no ambiguity about what is being applied to the circle."""
    ax.imshow(np.ones((2, 2, 3)), interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="#888888", linewidth=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f"{A[r, c]:g}", ha="center", va="center", fontsize=14)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    ax.set_title("matrix A", fontsize=12)
    ax.text(0.5, -0.14, "⎡x'⎤       ⎡x⎤",
             transform=ax.transAxes, ha="center", va="top", fontsize=12, family="monospace")
    ax.text(0.5, -0.26, "⎣y'⎦  =  A ⎣y⎦",
             transform=ax.transAxes, ha="center", va="top", fontsize=12, family="monospace")


def render(t, final_hold=False):
    """t: interpolation factor from 0 (identity, a plain circle) to 1 (A,
    the full ellipse). M(t) = (1-t) I + t A."""
    M = (1 - t) * np.eye(2) + t * A

    fig, (ax_mat, ax_main) = plt.subplots(
        1, 2, figsize=(9.6, 5.4), dpi=120,
        gridspec_kw={"width_ratios": [0.3, 1.0]},
    )
    fig.subplots_adjust(left=0.03, right=0.95, top=0.84, bottom=0.18, wspace=0.18)

    draw_matrix_panel(ax_mat)

    ax_main.set_xlim(-LIMIT, LIMIT)
    ax_main.set_ylim(-LIMIT, LIMIT)
    ticks = np.arange(-4, 4.1, 2)
    ax_main.set_xticks(ticks)
    ax_main.set_yticks(ticks)
    ax_main.set_aspect("equal")
    ax_main.set_autoscale_on(False)
    ax_main.axhline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax_main.axvline(0, color="#dddddd", linewidth=0.8, zorder=0)

    # Reference unit circle, always shown faint and dashed.
    ax_main.plot(CIRCLE[0], CIRCLE[1], linestyle="--", color="#cccccc", linewidth=1.0, zorder=1)

    # The circle morphing into the ellipse.
    curve = M @ CIRCLE
    ax_main.plot(curve[0], curve[1], color="#333333", linewidth=1.8, zorder=2)

    # Ordinary vectors: all but the two eigen-directions visibly rotate.
    tips = M @ arrow_starts
    for i in range(N_ARROWS):
        ax_main.annotate("", xy=(tips[0, i], tips[1, i]), xytext=(0, 0),
                          arrowprops=dict(arrowstyle="->", color=VEC_COLOR, lw=1.3), zorder=3)

    # Eigenvectors: stay on the same line the whole time, only their length
    # changes, scaling from 1 (t=0) to the eigenvalue (t=1).
    for v, lam, color, label in (
        (eigvecs[:, 0], eigvals[0], MAJOR_COLOR, "λ₁"),
        (eigvecs[:, 1], eigvals[1], MINOR_COLOR, "λ₂"),
    ):
        scale = (1 - t) + t * lam
        tip = v * scale
        ax_main.annotate("", xy=(tip[0], tip[1]), xytext=(0, 0),
                          arrowprops=dict(arrowstyle="->", color=color, lw=3.0), zorder=5)
        label_pt = v * (scale + 0.35)
        ax_main.text(label_pt[0], label_pt[1], label, color=color, fontsize=14,
                      ha="center", va="center", fontweight="bold", zorder=6)

    if t <= 0.0:
        suptitle = "a circle of vectors, before any transformation"
    elif t >= 1.0:
        suptitle = "most vectors rotate off their original direction..."
    else:
        suptitle = "apply A"

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, color="#222222")

    if final_hold:
        ax_main.text(0.5, -0.16,
                      "...except the eigenvectors (red, blue) - they only scaled by λ₁, λ₂",
                      transform=ax_main.transAxes, ha="center", va="top", fontsize=11, color="#444444")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_START = 14
HOLD_END = 36

frames += [render(0.0)] * HOLD_START

# Smooth morph from identity to A.
T_STEPS = np.linspace(0, 1, 36)[1:-1]
for t in T_STEPS:
    frames.append(render(t))

frames += [render(1.0, final_hold=True)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=70,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
