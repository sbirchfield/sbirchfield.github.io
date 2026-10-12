"""Generate gradient_descent_unsolvable_demo.gif: an animated companion to
the lesson 31 notebook's "Training on the unsolvable problem" cell. Trains
a single-neuron (logistic regression) classifier by gradient descent on the
notebook's own ring-vs-disk dataset, animating the decision boundary and
the loss curve epoch by epoch -- showing live that it converges to a
near-zero weight vector and roughly chance-level accuracy, worse even than
Lesson 30's brute-force 74% ceiling, because the symmetric dataset makes
the average gradient pull from all points nearly cancel out in every
direction.

Reuses the notebook's own forward/backward/train() logic and rng draw order
(class_a/class_b consumed first, exactly as in the notebook, before the
ring/disk draw) so the dataset matches exactly.

Run from anywhere:
    python make_gradient_descent_unsolvable_demo.py
Output:
    gradient_descent_unsolvable_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "gradient_descent_unsolvable_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"
ERROR_COLOR = "#c0392b"

# Same rng draw order as the notebook: class_a/class_b consumed first (cell 4),
# then the ring/disk dataset (cell 23), so `rng`'s state matches exactly.
rng = np.random.default_rng(1)
_ = rng.normal(loc=[-2, -1], scale=0.8, size=(60, 2))  # class_a (unused here)
_ = rng.normal(loc=[2, 1.5], scale=0.8, size=(60, 2))  # class_b (unused here)

theta_in = rng.uniform(0, 2 * np.pi, 60)
inner = np.stack([0.5 * np.cos(theta_in), 0.5 * np.sin(theta_in)], axis=1) + rng.normal(0, 0.1, (60, 2))
theta_out = rng.uniform(0, 2 * np.pi, 60)
outer = np.stack([2.0 * np.cos(theta_out), 2.0 * np.sin(theta_out)], axis=1) + rng.normal(0, 0.15, (60, 2))
X_ring = np.vstack([inner, outer])
y_ring = np.concatenate([np.zeros(60), np.ones(60)])

LESSON30_CEILING = 0.74


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(X, w, b):
    z = X @ w + b
    return sigmoid(z), z


def bce_loss(p, y, eps=1e-9):
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


def backward(X, p, y):
    grad_z = (p - y) / len(y)
    grad_w = X.T @ grad_z
    grad_b = grad_z.sum()
    return grad_w, grad_b


def plot_boundary(ax, w, b, title):
    ax.scatter(*inner.T, s=15, label="inner (disk)")
    ax.scatter(*outer.T, s=15, label="outer (ring)")
    w_norm = np.linalg.norm(w)
    if w_norm > 1e-6:
        w_hat = w / w_norm
        foot = -b * w_hat
        perp = np.array([-w_hat[1], w_hat[0]])
        p1, p2 = foot + 5 * perp, foot - 5 * perp
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black", linewidth=1.5, label="decision boundary")
        ax.arrow(*foot, *w_hat, head_width=0.15, color=ACTIVE_COLOR, length_includes_head=True)
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9.5)
    ax.legend(fontsize=7, loc="lower right")


def render(epoch, w, b, losses, n_epochs_total, loss_ylim):
    p, _ = forward(X_ring, w, b)
    acc = ((p > 0.5).astype(float) == y_ring).mean()
    loss = losses[-1]

    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.2), dpi=120)

    plot_boundary(axes[0], w, b, f"epoch {epoch:4d}   accuracy = {acc:.0%}   |w| = {np.linalg.norm(w):.3f}")

    ax2 = axes[1]
    ax2.plot(losses, color=NEUTRAL_COLOR, linewidth=1.3)
    ax2.axhline(-np.log(0.5), color="#999999", linewidth=1.0, linestyle=":")
    ax2.text(n_epochs_total * 0.03, -np.log(0.5), "chance-level loss  ", fontsize=8, color="#777777",
              ha="left", va="bottom")
    ax2.scatter([epoch], [loss], color=ACTIVE_COLOR, s=50, zorder=5)
    ax2.set_xlim(0, n_epochs_total)
    ax2.set_ylim(*loss_ylim)
    ax2.set_xlabel("epoch", fontsize=9)
    ax2.set_ylabel("loss", fontsize=9)
    ax2.set_title(f"training loss  (now: {loss:.4f}) -- barely moves: stuck near chance", fontsize=9.5)

    color = ERROR_COLOR if acc < LESSON30_CEILING - 0.03 else (MATCH_COLOR if acc >= LESSON30_CEILING else ACTIVE_COLOR)
    fig.suptitle(f"gradient descent on the unsolvable problem  "
                 f"(Lesson 30's brute-force ceiling: {LESSON30_CEILING:.0%})",
                 fontsize=11.5, color=color, fontweight="bold")
    fig.subplots_adjust(top=0.84, bottom=0.1, wspace=0.28)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# Train once, recording a checkpoint every few epochs (same train() logic/hyperparameters
# as the notebook's own train(X_ring, y_ring, n_epochs=2000) call).
rng_train = np.random.default_rng(0)
w = rng_train.normal(size=X_ring.shape[1]) * 0.1
b = 0.0
N_EPOCHS = 100
CHECKPOINT_EVERY = 20

losses = []
checkpoints = []  # (epoch, w copy, b, losses-so-far copy)
for epoch in range(N_EPOCHS):
    p, _ = forward(X_ring, w, b)
    losses.append(bce_loss(p, y_ring))
    if epoch % CHECKPOINT_EVERY == 0 or epoch == N_EPOCHS - 1:
        checkpoints.append((epoch, w.copy(), b, list(losses)))
    grad_w, grad_b = backward(X_ring, p, y_ring)
    w -= 0.5 * grad_w
    b -= 0.5 * grad_b

frames = []
HOLD_STEP = 2
HOLD_END = 11
_pad = (max(losses) - min(losses)) * 0.2
LOSS_YLIM = (min(losses) - _pad, max(losses) + _pad)

for epoch, w_snap, b_snap, losses_snap in checkpoints:
    frames += [render(epoch, w_snap, b_snap, losses_snap, N_EPOCHS, LOSS_YLIM)] * HOLD_STEP
frames += [frames[-1]] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=220,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames, final accuracy "
      f"{((sigmoid(X_ring @ w + b) > 0.5).astype(float) == y_ring).mean():.1%})")
