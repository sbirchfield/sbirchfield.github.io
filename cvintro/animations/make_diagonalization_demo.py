"""Generate diagonalization_demo.gif: an animated walkthrough of A = P Lambda P^T,
showing that applying A can be split into three understandable steps: rotate
the eigenvectors onto the axes (P^T), stretch along the axes by the
eigenvalues (Lambda), then rotate back (P). Companion to make_eigen_demo.py,
which shows the same matrix A applied directly in one step.

Run from anywhere:
    python make_diagonalization_demo.py
Output:
    diagonalization_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "diagonalization_demo.gif")

# Same matrix used in the lesson 6 notebook's eigenvector refresher and
# diagonalization cells.
A = np.array([[3.0, 1.0],
              [1.0, 1.5]])

eigvals, eigvecs = np.linalg.eigh(A)  # ascending order
order = [1, 0]  # index 0 = major (largest eigenvalue), index 1 = minor
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

P = eigvecs
Lam = np.diag(eigvals)

# P (and P^T) are proper rotations here (det = +1), but linearly
# interpolating their entries against the identity does NOT stay a proper
# rotation in between -- for a rotation this far from 0 degrees it can
# collapse the whole circle down toward a point partway through. Instead,
# interpolate the rotation ANGLE linearly, which stays well-conditioned the
# whole way.
THETA_PT = np.arctan2(P.T[1, 0], P.T[0, 0])


def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

MAJOR_COLOR = "#e54040"  # red: eigenvector with the larger eigenvalue
MINOR_COLOR = "#1f6fd6"  # blue: eigenvector with the smaller eigenvalue
VEC_COLOR = "#999999"    # gray: ordinary (non-eigen) vectors
ACTIVE_COLOR = "#e07b00"  # orange: highlights whichever matrix is being applied

N_ARROWS = 12
arrow_angles = np.linspace(0, 2 * np.pi, N_ARROWS, endpoint=False)
arrow_starts = np.stack([np.cos(arrow_angles), np.sin(arrow_angles)])  # 2xN

THETA = np.linspace(0, 2 * np.pi, 200)
CIRCLE = np.stack([np.cos(THETA), np.sin(THETA)])  # 2x200

LIMIT = eigvals[0] * 1.25


def M_at(s):
    """Cumulative transform at global phase s in [0, 3]: phase 1 (s in
    [0,1]) rotates by P^T, phase 2 (s in [1,2]) scales by Lambda, phase 3
    (s in [2,3]) rotates back by P. At s=3 this equals P @ Lambda @ P.T = A."""
    if s <= 1:
        t = s
        return rotation_matrix(t * THETA_PT)
    elif s <= 2:
        t = s - 1
        return ((1 - t) * np.eye(2) + t * Lam) @ rotation_matrix(THETA_PT)
    else:
        t = min(s - 2, 1.0)
        return rotation_matrix(-t * THETA_PT) @ Lam @ rotation_matrix(THETA_PT)


def draw_small_matrix(ax, M, title, active):
    """A small static 2x2 (or 1x2 label row) matrix panel. `active` draws a
    highlighted border, used to show which matrix is currently being
    applied to the circle."""
    ax.imshow(np.ones((2, 2, 3)), interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="#888888", linewidth=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for r in range(2):
        for c in range(2):
            val = M[r, c]
            text = "0" if val == 0 else f"{val:.2f}"
            ax.text(c, r, text, ha="center", va="center", fontsize=10)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    color = ACTIVE_COLOR if active else "#222222"
    weight = "bold" if active else "normal"
    ax.set_title(title, fontsize=12, color=color, fontweight=weight)
    if active:
        for spine in ax.spines.values():
            spine.set_edgecolor(ACTIVE_COLOR)
            spine.set_linewidth(3.0)


def render(s, final_hold=False):
    M = M_at(s)
    phase1 = s <= 1.0
    phase2 = 1.0 < s <= 2.0
    phase3 = s > 2.0

    fig, (ax_p, ax_lam, ax_pt, ax_main) = plt.subplots(
        1, 4, figsize=(13.0, 5.0), dpi=120,
        gridspec_kw={"width_ratios": [0.26, 0.26, 0.26, 1.0]},
    )
    fig.subplots_adjust(left=0.02, right=0.97, top=0.82, bottom=0.18, wspace=0.35)

    draw_small_matrix(ax_p, P, "P", phase3)
    draw_small_matrix(ax_lam, Lam, "Λ", phase2)
    draw_small_matrix(ax_pt, P.T, "Pᵀ", phase1)

    fig.text(0.22, 0.14, "A = P Λ Pᵀ", fontsize=14, ha="center", va="center", color="#222222")

    ax_main.set_xlim(-LIMIT, LIMIT)
    ax_main.set_ylim(-LIMIT, LIMIT)
    ticks = np.arange(-4, 4.1, 2)
    ax_main.set_xticks(ticks)
    ax_main.set_yticks(ticks)
    ax_main.set_aspect("equal")
    ax_main.set_autoscale_on(False)
    ax_main.axhline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax_main.axvline(0, color="#dddddd", linewidth=0.8, zorder=0)

    ax_main.plot(CIRCLE[0], CIRCLE[1], linestyle="--", color="#cccccc", linewidth=1.0, zorder=1)

    curve = M @ CIRCLE
    ax_main.plot(curve[0], curve[1], color="#333333", linewidth=1.8, zorder=2)

    tips = M @ arrow_starts
    for i in range(N_ARROWS):
        ax_main.annotate("", xy=(tips[0, i], tips[1, i]), xytext=(0, 0),
                          arrowprops=dict(arrowstyle="->", color=VEC_COLOR, lw=1.3), zorder=3)

    for v, color, label in (
        (eigvecs[:, 0], MAJOR_COLOR, "v₁"),
        (eigvecs[:, 1], MINOR_COLOR, "v₂"),
    ):
        tip = M @ v
        ax_main.annotate("", xy=(tip[0], tip[1]), xytext=(0, 0),
                          arrowprops=dict(arrowstyle="->", color=color, lw=3.0), zorder=5)
        norm = np.linalg.norm(tip)
        label_pt = tip / max(norm, 1e-6) * (norm + 0.35)
        ax_main.text(label_pt[0], label_pt[1], label, color=color, fontsize=14,
                      ha="center", va="center", fontweight="bold", zorder=6)

    if s <= 0.0:
        suptitle = "start: circle of vectors, eigenvectors v₁, v₂ shown"
    elif phase1:
        suptitle = "step 1: apply Pᵀ — rotate so eigenvectors align with the axes"
    elif phase2:
        suptitle = "step 2: apply Λ — stretch along the axes by the eigenvalues"
    elif s < 3.0:
        suptitle = "step 3: apply P — rotate back"
    else:
        suptitle = ""

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, color="#222222")

    if final_hold:
        ax_main.text(0.5, -0.14, "A = P Λ Pᵀ — same ellipse as applying A directly",
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

S_STEPS = np.linspace(0, 3, 60)[1:-1]
for s in S_STEPS:
    frames.append(render(s))

frames += [render(3.0, final_hold=True)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=100,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
