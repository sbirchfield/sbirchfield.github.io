"""Generate gradient_descent_1d_demo.gif: an animated companion to the
lesson 31 notebook's "Gradient descent: using the slope to find a minimum"
cell. Shows gradient descent on f(x) = (x-3)^2 + 1 one step at a time: at
each frame the tangent line at the current point makes the slope f'(x)
visible directly, an arrow shows the step taken against that slope, and the
point lands at its new position -- literally "follow the slope downhill."

Run from anywhere:
    python make_gradient_descent_1d_demo.py
Output:
    gradient_descent_1d_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "gradient_descent_1d_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"


def f(x):
    return (x - 3) ** 2 + 1


def fprime(x):
    return 2 * (x - 3)


LR = 0.3
N_STEPS = 10
xs = np.linspace(-5, 8, 200)

path = [-4.0]
for _ in range(N_STEPS):
    path.append(path[-1] - LR * fprime(path[-1]))
path = np.array(path)


def render(step, show_tangent):
    x = path[step]
    fig, ax = plt.subplots(figsize=(6.0, 4.4), dpi=120)
    ax.plot(xs, f(xs), color="#aaaaaa", linewidth=1.8, zorder=1)

    # Already-visited points, faint trail.
    ax.plot(path[:step + 1], f(path[:step + 1]), "o-", color=ACTIVE_COLOR,
             markersize=5, linewidth=1.1, alpha=0.55, zorder=2)

    slope = fprime(x)
    if show_tangent:
        tx = np.array([x - 2.2, x + 2.2])
        ty = f(x) + slope * (tx - x)
        ax.plot(tx, ty, color=NEUTRAL_COLOR, linewidth=1.6, linestyle="--", zorder=3)
        step_x = x - LR * slope
        ax.annotate("", xy=(step_x, f(x)), xytext=(x, f(x)),
                     arrowprops=dict(arrowstyle="->", color=ACTIVE_COLOR, lw=2.2), zorder=5)

    dot_color = MATCH_COLOR if abs(slope) < 0.15 else ACTIVE_COLOR
    ax.scatter([x], [f(x)], color=dot_color, s=90, zorder=6, edgecolors="white", linewidths=0.8)

    ax.set_xlim(-5.5, 8.5)
    ax.set_ylim(-3, 68)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    verdict = "slope ~ 0: at the minimum" if abs(slope) < 0.15 else f"slope = {slope:+.2f}: step against it"
    ax.set_title(f"step {step}/{N_STEPS}   x = {x:.2f}\n{verdict}", fontsize=11, color=dot_color)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_START = 14
HOLD_STEP = 10
HOLD_END = 36

frames += [render(0, show_tangent=False)] * HOLD_START
for step in range(N_STEPS):
    frames += [render(step, show_tangent=True)] * HOLD_STEP
    frames += [render(step + 1, show_tangent=False)] * 2
frames += [render(N_STEPS, show_tangent=False)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=90,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
