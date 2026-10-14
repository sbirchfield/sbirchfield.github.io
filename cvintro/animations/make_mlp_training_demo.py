"""Generate mlp_training_demo.gif: an animated companion to the lesson 32
notebook's "Training" / "The hidden layer's projections, made visible"
cells. Trains a 2-layer MLP (2 -> 4 ReLU -> 1) on the notebook's own
ring-vs-disk dataset, animating three things together as training
progresses: each hidden unit's own straight-line cut (same style as the
notebook's own hidden-unit-lines figure), the resulting *combined* decision
region (shaded, from thresholding the full network's output on a grid), and
the loss curve -- showing live how several straight lines, combined through
one more linear layer, bend into a closed curve that successfully wraps
around the ring, succeeding where Lesson 31's single neuron could not.

Reuses the notebook's own forward/backward/train hyperparameters and rng
seeds (data: default_rng(0); weights: default_rng(8)) so the trained model
matches the static cells exactly.

Run from anywhere:
    python make_mlp_training_demo.py
Output:
    mlp_training_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "mlp_training_demo.gif")

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


# Same init as the notebook's "Backprop through two layers" cell.
H = 4
init_rng = np.random.default_rng(8)
W1 = init_rng.normal(size=(2, H)) * 0.7
b1 = np.zeros(H)
W2 = init_rng.normal(size=(H, 1)) * 0.7
b2 = np.zeros(1)

LR = 0.1
N_EPOCHS = 3000
CHECKPOINT_EVERY = 15

GRID_LIM = 3.0
gx, gy = np.meshgrid(np.linspace(-GRID_LIM, GRID_LIM, 160), np.linspace(-GRID_LIM, GRID_LIM, 160))
grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

# Red, green, purple, yellow -- avoids the blue/orange already used for the
# inner/outer data points.
HIDDEN_COLORS = ["#d62728", "#2ca02c", "#9467bd", "#dbc200"]


def render(epoch, W1, b1, W2, b2, losses, loss_ylim):
    p, _ = forward(X, W1, b1, W2, b2)
    acc = ((p > 0.5).astype(float) == y).mean()

    grid_p, _ = forward(grid_pts, W1, b1, W2, b2)
    region = (grid_p > 0.5).reshape(gx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), dpi=120)

    ax = axes[0]
    ax.contourf(gx, gy, region, levels=[-0.5, 0.5, 1.5],
                colors=["#a9cdf5", "#fde0c4"], alpha=0.65, zorder=1)
    points_handle_in = ax.scatter(*inner.T, s=15, label="inner points", zorder=3)
    points_handle_out = ax.scatter(*outer.T, s=15, label="outer points", zorder=3)
    unit_handles = []
    for i in range(H):
        w_i, b_i = W1[:, i], b1[i]
        w_norm2 = w_i @ w_i
        if w_norm2 < 1e-9:
            unit_handles.append(None)
            continue
        foot = -b_i * w_i / w_norm2
        perp = np.array([-w_i[1], w_i[0]])
        p1, p2 = foot + 5 * perp, foot - 5 * perp
        line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=HIDDEN_COLORS[i], linewidth=1.3,
                          alpha=0.85, zorder=2, label=f"hidden unit {i}")
        unit_handles.append(line)
    ax.set_xlim(-GRID_LIM, GRID_LIM); ax.set_ylim(-GRID_LIM, GRID_LIM)
    ax.set_aspect("equal")
    # Two separate legends -- data points on the left, hidden-unit lines on
    # the right -- rather than one legend ncol=2 (which splits the 6 entries
    # 3/3 instead of grouping them by what they are).
    points_legend = ax.legend(handles=[points_handle_in, points_handle_out],
                                fontsize=6.5, loc="upper left")
    ax.add_artist(points_legend)
    ax.legend(handles=[h for h in unit_handles if h is not None],
              fontsize=6.5, loc="upper right")
    ax.set_title(f"epoch {epoch:4d}   accuracy = {acc:.0%}", fontsize=10)

    ax2 = axes[1]
    ax2.plot(losses, color=NEUTRAL_COLOR, linewidth=1.3)
    ax2.scatter([epoch], [losses[-1]], color=ACTIVE_COLOR, s=50, zorder=5)
    ax2.set_xlim(0, N_EPOCHS)
    ax2.set_ylim(*loss_ylim)
    ax2.set_xlabel("epoch", fontsize=9)
    ax2.set_ylabel("loss", fontsize=9)
    ax2.set_title(f"training loss  (now: {losses[-1]:.3f})", fontsize=10)

    color = MATCH_COLOR if acc >= 0.97 else (ACTIVE_COLOR if acc >= 0.74 else NEUTRAL_COLOR)
    fig.suptitle("several straight cuts + one linear layer = polygonal boundary",
                 fontsize=11.5, color=color, fontweight="bold")
    fig.subplots_adjust(top=0.86, bottom=0.1, wspace=0.26)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


losses = []
checkpoints = []
for epoch in range(N_EPOCHS):
    p, cache = forward(X, W1, b1, W2, b2)
    eps = 1e-9
    losses.append(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    if epoch % CHECKPOINT_EVERY == 0 or epoch == N_EPOCHS - 1:
        checkpoints.append((epoch, W1.copy(), b1.copy(), W2.copy(), b2.copy(), list(losses)))
    grad_W1, grad_b1, grad_W2, grad_b2 = backward(X, y, p, cache, W2)
    W1 -= LR * grad_W1; b1 -= LR * grad_b1
    W2 -= LR * grad_W2; b2 -= LR * grad_b2

frames = []
HOLD_STEP = 2
HOLD_END = 50
LOSS_YLIM = (0, max(losses) * 1.08)

for epoch, W1s, b1s, W2s, b2s, losses_snap in checkpoints:
    frames += [render(epoch, W1s, b1s, W2s, b2s, losses_snap, LOSS_YLIM)] * HOLD_STEP
frames += [frames[-1]] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=55,
    loop=0,
)
final_p, _ = forward(X, W1, b1, W2, b2)
print(f"Saved {OUT_GIF} ({len(frames)} frames, final accuracy "
      f"{((final_p > 0.5).astype(float) == y).mean():.1%})")
