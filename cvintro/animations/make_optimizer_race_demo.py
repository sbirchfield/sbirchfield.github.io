"""Generate optimizer_race_demo.gif: an animated companion to the lesson 33
notebook's "All three paths, visualized" cell. Races plain SGD, SGD with
momentum, and Adam step by step on the same narrow-bowl loss landscape,
starting from the same point with the same per-step learning rates as the
notebook -- animating the static "three paths" comparison into a live race
so the characteristic failure/success modes are visible as they happen:
plain SGD zigzagging slowly across the narrow valley, momentum overshooting
past the minimum before settling, and Adam adapting its per-parameter step
size to make steady progress down the shallow axis while damping the steep
one.

Reuses the notebook's own f/grad_f, hyperparameters, and update rules
exactly, so the final losses match the static cells.

Run from anywhere:
    python make_optimizer_race_demo.py
Output:
    optimizer_race_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "optimizer_race_demo.gif")

SGD_COLOR = "tab:blue"
MOM_COLOR = "#e07b00"
ADAM_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"


def f(p):
    return 0.05 * p[0] ** 2 + 5 * p[1] ** 2


def grad_f(p):
    return np.array([0.1 * p[0], 10 * p[1]])


start = np.array([-4.0, 1.0])
N_STEPS = 30

# Plain SGD -- same lr as the notebook.
lr = 0.15
p = start.copy()
sgd_path = [p.copy()]
for _ in range(N_STEPS):
    p = p - lr * grad_f(p)
    sgd_path.append(p.copy())
sgd_path = np.array(sgd_path)

# SGD + momentum -- same hyperparameters as the notebook.
momentum = 0.9
p = start.copy()
v = np.zeros(2)
mom_path = [p.copy()]
for _ in range(N_STEPS):
    g = grad_f(p)
    v = momentum * v - lr * g
    p = p + v
    mom_path.append(p.copy())
mom_path = np.array(mom_path)

# Adam -- same hyperparameters as the notebook.
beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
adam_lr = 0.3
p = start.copy()
m, v_sq = np.zeros(2), np.zeros(2)
adam_path = [p.copy()]
for t in range(1, N_STEPS + 1):
    g = grad_f(p)
    m = beta1 * m + (1 - beta1) * g
    v_sq = beta2 * v_sq + (1 - beta2) * g ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v_sq / (1 - beta2 ** t)
    p = p - adam_lr * m_hat / (np.sqrt(v_hat) + adam_eps)
    adam_path.append(p.copy())
adam_path = np.array(adam_path)

xs = np.linspace(-4.5, 1, 200)
ys = np.linspace(-1.5, 1.5, 200)
XX, YY = np.meshgrid(xs, ys)
ZZ = 0.05 * XX ** 2 + 5 * YY ** 2

PATHS = [("plain SGD", sgd_path, SGD_COLOR), ("momentum", mom_path, MOM_COLOR), ("Adam", adam_path, ADAM_COLOR)]


def render(step):
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=120)
    ax.contour(XX, YY, ZZ, levels=20, colors="lightgray", linewidths=0.7)
    ax.scatter([0], [0], marker="*", s=180, color="black", zorder=6)

    for name, path, color in PATHS:
        i = min(step, len(path) - 1)
        ax.plot(*path[:i + 1].T, "-o", markersize=3, color=color, linewidth=1.5, zorder=4,
                 label=f"{name}  (loss={f(path[i]):.3f})")
        ax.scatter(*path[i], s=90, color=color, zorder=5, edgecolors="black", linewidths=0.8)

    ax.set_xlim(-4.7, 1.2); ax.set_ylim(-1.6, 1.6)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(f"step {min(step, N_STEPS):2d} / {N_STEPS}", fontsize=10.5)
    fig.suptitle("racing three optimizers down the same narrow bowl",
                 fontsize=12.5, color=NEUTRAL_COLOR, fontweight="bold")
    fig.subplots_adjust(top=0.88, bottom=0.08)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 5
HOLD_END = 48

for step in range(N_STEPS + 1):
    frames += [render(step)] * HOLD_STEP
frames += [frames[-1]] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=70,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
print(f"final losses: SGD={f(sgd_path[-1]):.4f}  momentum={f(mom_path[-1]):.4f}  Adam={f(adam_path[-1]):.4f}")
