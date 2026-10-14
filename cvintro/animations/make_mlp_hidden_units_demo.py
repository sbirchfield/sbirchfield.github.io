"""Generate mlp_hidden_units_demo.gif: an animated companion to the lesson 32
notebook's "The hidden layer's projections, made visible" section, shown
right after the mlp_training_demo.gif training animation. Trains the same
2-layer MLP (2 -> 4 ReLU -> 1) to convergence on the notebook's own
ring-vs-disk dataset, then freezes the weights and steps through the 4
hidden units one at a time, side by side:

  - Left panel: just one hidden unit at a time -- its cut (line + direction
    arrow) and the half-plane where it's "on" (ReLU output > 0), shaded in
    that unit's own color. The other three units are drawn faint/grey.
  - Right panel: the *cumulative* effect of the units introduced so far --
    a real partial forward pass through the (frozen) output layer, with the
    not-yet-introduced units' contributions zeroed out of the sum, not a
    simplified intersection of half-planes. By the last step all 4 units are
    included and this panel exactly matches the network's real decision
    boundary.

Both panels advance through the same 4 steps in lockstep, so it reads as
"here's unit i alone" (left) immediately next to "here's what we have after
adding unit i" (right).

Reuses the notebook's own forward/backward/train hyperparameters and rng
seeds (data: default_rng(0); weights: default_rng(8)) so the trained model
matches the static cells and the training animation exactly.

Run from anywhere:
    python make_mlp_hidden_units_demo.py
Output:
    mlp_hidden_units_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "mlp_hidden_units_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"

# Same data as the notebook's "The unsolvable problem, again" cell.
rng = np.random.default_rng(0)
theta_in = rng.uniform(0, 2 * np.pi, 60)
inner = np.stack([0.5 * np.cos(theta_in), 0.5 * np.sin(theta_in)], axis=1) + rng.normal(0, 0.1, (60, 2))
theta_out = rng.uniform(0, 2 * np.pi, 60)
outer = np.stack([2.0 * np.cos(theta_out), 2.0 * np.sin(theta_out)], axis=1) + rng.normal(0, 0.15, (60, 2))
X = np.vstack([inner, outer])
y = np.concatenate([np.zeros(60), np.ones(60)])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def relu_deriv(z):
    return (z > 0).astype(np.float64)


def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = (a1 @ W2 + b2).ravel()
    p = sigmoid(z2)
    return p, (z1, a1, z2)


def backward(X, y, p, cache, W2):
    z1, a1, z2 = cache
    n = len(y)
    grad_z2 = ((p - y) / n).reshape(-1, 1)
    grad_W2 = a1.T @ grad_z2
    grad_b2 = grad_z2.sum(axis=0)
    grad_a1 = grad_z2 @ W2.T
    grad_z1 = grad_a1 * relu_deriv(z1)
    grad_W1 = X.T @ grad_z1
    grad_b1 = grad_z1.sum(axis=0)
    return grad_W1, grad_b1, grad_W2, grad_b2


# Same init as the notebook's "Backprop through two layers" cell, trained to
# convergence exactly as the "Training" cell does.
H = 4
init_rng = np.random.default_rng(8)
W1 = init_rng.normal(size=(2, H)) * 0.7
b1 = np.zeros(H)
W2 = init_rng.normal(size=(H, 1)) * 0.7
b2 = np.zeros(1)

LR = 0.1
N_EPOCHS = 3000
for _ in range(N_EPOCHS):
    p, cache = forward(X, W1, b1, W2, b2)
    grad_W1, grad_b1, grad_W2, grad_b2 = backward(X, y, p, cache, W2)
    W1 -= LR * grad_W1; b1 -= LR * grad_b1
    W2 -= LR * grad_W2; b2 -= LR * grad_b2

final_p, _ = forward(X, W1, b1, W2, b2)
print(f"Converged model accuracy: {((final_p > 0.5).astype(float) == y).mean():.1%}")

GRID_LIM = 3.0
gx, gy = np.meshgrid(np.linspace(-GRID_LIM, GRID_LIM, 160), np.linspace(-GRID_LIM, GRID_LIM, 160))
grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
grid_z1 = grid_pts @ W1 + b1          # (n_grid, H), reused every frame -- weights are frozen
grid_a1 = relu(grid_z1)

# Red, green, purple, yellow -- avoids the blue/orange already used for the
# inner/outer data points.
HIDDEN_COLORS = ["#d62728", "#2ca02c", "#9467bd", "#dbc200"]


def unit_line_endpoints(i):
    w_i, b_i = W1[:, i], b1[i]
    foot = -b_i * w_i / (w_i @ w_i)
    perp = np.array([-w_i[1], w_i[0]])
    return foot + 5 * perp, foot - 5 * perp, foot, w_i / np.linalg.norm(w_i)


def draw_scatter(ax):
    ax.scatter(*inner.T, s=14, label="inner points", zorder=3)
    ax.scatter(*outer.T, s=14, label="outer points", zorder=3)


def render(k):
    """k = number of hidden units introduced so far (1..H)."""
    current = k - 1

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), dpi=120)

    # Left: unit `current` alone -- its active half-plane shaded, its line
    # highlighted, the other three units drawn faint for context only.
    ax = axes[0]
    active = (grid_z1[:, current] > 0).reshape(gx.shape)
    ax.contourf(gx, gy, active, levels=[-0.5, 0.5, 1.5],
                colors=["#ffffff", HIDDEN_COLORS[current]], alpha=0.30, zorder=1)
    draw_scatter(ax)
    for i in range(H):
        p1, p2, foot, direction = unit_line_endpoints(i)
        if i == current:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=HIDDEN_COLORS[i],
                     linewidth=2.2, zorder=4, label=f"hidden unit {i}")
            ax.arrow(*foot, *(direction * 0.5), head_width=0.12, color=HIDDEN_COLORS[i],
                      length_includes_head=True, zorder=5)
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#bbbbbb",
                     linewidth=1.0, alpha=0.7, zorder=2)
    ax.set_xlim(-GRID_LIM, GRID_LIM); ax.set_ylim(-GRID_LIM, GRID_LIM)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")
    ax.text(0.5, 1.22, "one hidden unit at a time", transform=ax.transAxes,
             ha="center", fontsize=11.5, fontweight="bold")
    ax.set_title(f"hidden unit {current} alone\n(shaded = ReLU active, i.e., > 0)", fontsize=10)

    # Right: real partial forward pass -- units >= k zeroed out of the sum,
    # so this is the network's actual output with only k units "turned on",
    # not a stand-in intersection of half-planes.
    ax2 = axes[1]
    masked_a1 = grid_a1.copy()
    masked_a1[:, k:] = 0.0
    grid_p_partial = sigmoid((masked_a1 @ W2 + b2).ravel()).reshape(gx.shape)
    region = grid_p_partial > 0.5
    ax2.contourf(gx, gy, region, levels=[-0.5, 0.5, 1.5],
                 colors=["#a9cdf5", "#fde0c4"], alpha=0.65, zorder=1)
    draw_scatter(ax2)
    for i in range(k):
        p1, p2, _, _ = unit_line_endpoints(i)
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], color=HIDDEN_COLORS[i],
                  linewidth=1.5, alpha=0.9, zorder=2)
    ax2.set_xlim(-GRID_LIM, GRID_LIM); ax2.set_ylim(-GRID_LIM, GRID_LIM)
    ax2.set_aspect("equal")

    masked_a1_train = relu(X @ W1 + b1)
    masked_a1_train[:, k:] = 0.0
    masked_p_train = sigmoid((masked_a1_train @ W2 + b2).ravel())
    acc_partial = ((masked_p_train > 0.5).astype(float) == y).mean()
    color = MATCH_COLOR if k == H else NEUTRAL_COLOR
    ax2.text(0.5, 1.22, "their combined effect", transform=ax2.transAxes,
              ha="center", fontsize=11.5, fontweight="bold")
    ax2.set_title(f"units 0..{current} combined\n(partial-sum accuracy = {acc_partial:.0%})",
                   fontsize=10, color=color)

    fig.subplots_adjust(top=0.78, bottom=0.08, wspace=0.22)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 55
HOLD_END = 70

for k in range(1, H + 1):
    frames += [render(k)] * (HOLD_END if k == H else HOLD_STEP)

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=60,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
