"""Generate projection_as_3d_ray_demo.gif: a step-by-step companion to the
lesson 30 notebook's "Matrix multiplication: many projections at once" cell,
making the single-row case geometric. The 3-element kernel [0.25, 0.5, 0.25]
is itself a direction in 3D; any 3-element window of the signal is a point
in that same 3D space; projecting that point onto the kernel's direction
(dropping a perpendicular from the point to the line through the origin)
lands at a signed distance along the line -- and that distance, once
rescaled by the direction's own length, is exactly y_i = w . x, the row's
dot product. Uses row 0's window exactly, matching the earlier
matrix-as-projections animation's first step (y_0 = 1.201).

Run from anywhere:
    python make_projection_as_3d_ray_demo.py
Output:
    projection_as_3d_ray_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "projection_as_3d_ray_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"
RAY_COLOR = "tab:blue"

# Same data as the matrix-as-projections animation's row 0: the kernel as a
# 3D direction, and the signal's first 3-element window as a 3D point.
w = np.array([0.25, 0.5, 0.25])
P = np.array([1.82843024, 2.02007337, -1.06477104])  # signal[0:3]

w_norm = np.linalg.norm(w)
w_hat = w / w_norm
s = P @ w_hat          # signed Euclidean distance along the ray (unit direction)
foot = s * w_hat        # P's orthogonal projection onto the ray
y0 = w @ P              # = w_norm * s -- the actual matrix-multiply output


def render(stage):
    fig = plt.figure(figsize=(7.6, 6.6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    # The ray: the line through the origin along w's direction.
    ray_a, ray_b = -1.0 * w_hat, 3.0 * w_hat
    ax.plot(*zip(ray_a, ray_b), color=RAY_COLOR, linewidth=2, zorder=2)
    ax.scatter(*w, color=RAY_COLOR, s=70, zorder=6, edgecolors="black", linewidths=0.7)
    ax.text(*(0.55 * w_hat + np.array([0.0, 0.0, 0.07])), "direction $w$",
             fontsize=9.5, color=RAY_COLOR, ha="right")

    # A small xyz reference triad, centered at the actual origin, so it
    # reads as 3D space even with the background panes/ticks switched off.
    triad_len = 0.4
    for vec, label in [((1, 0, 0), "x"), ((0, 1, 0), "y"), ((0, 0, 1), "z")]:
        tip = np.array(vec) * triad_len
        ax.plot(*zip([0, 0, 0], tip), color="#777777", linewidth=1.2, zorder=1)
        ax.text(*(np.array(vec) * triad_len * 1.3), label, fontsize=9, color="#777777")
    ax.scatter(*[0, 0, 0], color=NEUTRAL_COLOR, s=40, zorder=4)
    ax.text(0.05, 0.05, -0.18, "origin", fontsize=8.5, color=NEUTRAL_COLOR, ha="right")

    if stage >= 1:
        ax.scatter(*P, color=ACTIVE_COLOR, s=90, zorder=6, edgecolors="black", linewidths=0.7)
        ax.text(*(P + np.array([-0.1, -0.1, 0.07])),
                 f"P = (x₀,x₁,x₂)\n= ({P[0]:.2f}, {P[1]:.2f}, {P[2]:.2f})",
                 fontsize=9.5, color=ACTIVE_COLOR, ha="right")

    if stage >= 2:
        ax.plot(*zip(P, foot), color="#999999", linewidth=1.3, linestyle="--", zorder=3)

    if stage >= 3:
        ax.scatter(*foot, color=MATCH_COLOR, s=90, zorder=6, edgecolors="black", linewidths=0.7)
        ax.plot(*zip([0, 0, 0], foot), color=MATCH_COLOR, linewidth=2.8, zorder=5)
        ax.text(*(foot + np.array([-0.1, -0.35, 0.15])),
                 f"foot of P on the ray\nsigned distance s = {s:.3f}",
                 fontsize=9, color=MATCH_COLOR)

    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.5, 2.5); ax.set_zlim(-1.5, 1.5)
    ax.set_axis_off()
    ax.view_init(elev=16, azim=-55)

    # Each stage's line stays on screen (stacked below the previous ones)
    # instead of replacing it, so the story builds up and nothing flashes by.
    lines = [
        ("kernel $w$ = [0.25, 0.5, 0.25] is a direction in 3D", RAY_COLOR),
        ("3-element window P=(x₀, x₁, x₂) of the signal is a point in the same 3D space", ACTIVE_COLOR),
        ("projecting P onto the ray: drop a perpendicular", NEUTRAL_COLOR),
        (f"y₀ = w·x = wᵀx = |w|s = {w_norm:.3f} · {s:.3f} = {y0:.3f}", MATCH_COLOR),
    ]
    for i in range(stage + 1):
        text, color = lines[i]
        current = i == stage
        fig.text(0.5, 0.985 - i * 0.045, text, ha="center", va="top",
                  fontsize=11.5 if current else 9.5,
                  color=color if current else "#aaaaaa",
                  fontweight="bold" if current else "normal")
    fig.subplots_adjust(top=1.55, bottom=-0.65, left=-0.15, right=1.15)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
STAGE_HOLD = [55, 55, 55, 70]  # frames to hold each of the 4 steps

for stage, hold in enumerate(STAGE_HOLD):
    frames += [render(stage)] * hold

# Crop the wasted white space below the lowest content (the orange point P)
# -- computed once across all distinct stages, then applied uniformly to
# every frame so the GIF's frame size stays consistent.
bottom_rows = []
for stage in range(len(STAGE_HOLD)):
    arr = np.array(render(stage).convert("L"))
    nonwhite_rows = np.where((arr < 250).any(axis=1))[0]
    bottom_rows.append(nonwhite_rows.max())
crop_bottom = min(max(bottom_rows) + 20, frames[0].height)
frames = [f.crop((0, 0, f.width, crop_bottom)) for f in frames]

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=55,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
