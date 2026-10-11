"""Generate vanishing_point_demo.gif: an animated version of the lesson 24
notebook's "Parallel lines stop being parallel" / "Vanishing points, computed
two ways" cells. Sweeps a homography H from the identity (parallelism intact)
to the notebook's own H (parallel lines converging), using the exact same
two vertical lines (the two rails of a train track) and H matrix as the
notebook. At every frame, the "vanishing point" is computed two ways --
intersecting the current transformed lines directly, and applying the
current (partial) homography to the point at infinity (0, 1, 0) -- and the
two stay locked together the whole time, not just at the end, since
H_alpha @ infinity = intersection of H_alpha @ line1 and H_alpha @ line2 for
ANY homography H_alpha.

Run from anywhere:
    python make_vanishing_point_demo.py
Output:
    vanishing_point_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "vanishing_point_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#333333"
DIM_COLOR = "#666666"
LINE1_COLOR = "tab:blue"
LINE2_COLOR = "tab:orange"

# Same H and lines as the notebook's "Parallel lines stop being parallel" cell.
H = np.array([
    [-3, 0,  2],
    [0,  2, -1],
    [0,  1,  1],
], dtype=np.float64)

line1 = np.array([[0, 0], [0, 1]], dtype=np.float64)   # x = 0
line2 = np.array([[1, 0], [1, 1]], dtype=np.float64)   # x = 1, parallel to line1

I3 = np.eye(3)


def apply_homography(points, H):
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    transformed = (H @ homogeneous.T).T
    return transformed[:, :2] / transformed[:, 2:3]


def line_intersection(p1, p2, p3, p4):
    A = np.array([[p2[0] - p1[0], -(p4[0] - p3[0])],
                  [p2[1] - p1[1], -(p4[1] - p3[1])]])
    b = np.array([p3[0] - p1[0], p3[1] - p1[1]])
    t = np.linalg.solve(A, b)[0]
    return p1 + t * (p2 - p1)


def extended(p1, p2, back, fwd):
    direction = p2 - p1
    return p1 - back * direction, p2 + fwd * direction


def fmt_vp(vp):
    return f"({vp[0]:6.2f}, {vp[1]:6.2f})"


def render(alpha):
    H_a = (1 - alpha) * I3 + alpha * H
    l1t = apply_homography(line1, H_a)
    l2t = apply_homography(line2, H_a)

    point_at_infinity = np.array([0.0, 1.0, 0.0])
    transformed_infinity = H_a @ point_at_infinity
    w = transformed_infinity[2]
    has_vp = abs(w) > 1e-4
    if has_vp:
        vp_via_infinity = transformed_infinity[:2] / w
        vp_direct = line_intersection(l1t[0], l1t[1], l2t[0], l2t[1])

    fig = plt.figure(figsize=(9.2, 5.4), dpi=120)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.62], left=0.03, right=0.98,
                            top=0.94, bottom=0.05, wspace=0.1)
    ax = fig.add_subplot(gs[0])
    ax_side = fig.add_subplot(gs[1])

    for line, color in [(l1t, LINE1_COLOR), (l2t, LINE2_COLOR)]:
        a, b = extended(line[0], line[1], back=60, fwd=60)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=1.8)
        ax.plot(line[:, 0], line[:, 1], 'o', color=color, zorder=3)
    if has_vp and np.all(np.abs(vp_via_infinity) < 1e4):
        ax.scatter(*vp_via_infinity, s=140, marker='*', color=ACTIVE_COLOR,
                    zorder=5, edgecolors="white", linewidths=1.0)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title("two originally-parallel lines, warped by $H_\\alpha$", fontsize=11)

    ax_side.axis("off")
    ax_side.set_xlim(0, 1)
    ax_side.set_ylim(0, 1)
    lines = [
        ("vanishing point", 17, NEUTRAL_COLOR, "bold", "sans-serif"),
        (f"$\\alpha$ = {alpha:4.2f}   ($H_\\alpha = (1-\\alpha)I + \\alpha H$)", 12.5, DIM_COLOR, "normal", "sans-serif"),
        ("point at infinity (0,1,0) ↦", 13, NEUTRAL_COLOR, "normal", "sans-serif"),
    ]
    if has_vp and np.all(np.abs(vp_via_infinity) < 1e4):
        lines += [
            (f"Hα·(0,1,0) = {fmt_vp(vp_via_infinity)}", 13, ACTIVE_COLOR, "normal", "monospace"),
            ("line intersection =", 13, NEUTRAL_COLOR, "normal", "sans-serif"),
            (f"{fmt_vp(vp_direct)}", 13, MATCH_COLOR, "normal", "monospace"),
            ("the two always agree ✓", 12.5, MATCH_COLOR, "bold", "sans-serif"),
        ]
    else:
        lines += [
            ("(still parallel —", 13, DIM_COLOR, "normal", "sans-serif"),
            (" vanishing point at infinity)", 13, DIM_COLOR, "normal", "sans-serif"),
        ]
    y = 0.90
    for text, fontsize, color, weight, family in lines:
        ax_side.text(0.0, y, text, fontsize=fontsize, color=color, fontweight=weight,
                      family=family, ha="left", va="top")
        y -= 0.085 * (text.count("\n") + 1) * (fontsize / 13.0) + 0.035

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


N = 36
HOLD_START = 14
HOLD_END = 50

alphas = np.linspace(0.0, 1.0, N + 1)

frames = [render(0.0)] * HOLD_START
for a in alphas[1:]:
    frames.append(render(a))
frames += [render(1.0)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=80,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
