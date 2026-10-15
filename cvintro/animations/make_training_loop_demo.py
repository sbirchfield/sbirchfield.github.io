"""Generate training_loop_demo.gif: an animated companion to the lesson 31
notebook's "Training: the full loop" section. Lesson 31 states the 4-step
training recipe in prose (forward pass, loss, backward pass, update) but
never shows it as a picture. This animation draws the single neuron as a
small computation graph: boxes are *operations* (a linear combination, a
sigmoid, a loss), and variables (x1, x2, the constant bias input, w1, w2, b,
z, p, y, L) are plain text labels riding along the arrows between them, not
shapes of their own. It cycles through the 4 stages, highlighting a
different part of the graph (thicker lines/borders, a moving dot along the
active arrows, and a different color) each time:

  1. forward  -- values flow left to right: inputs -> [w.x + b] -> z ->
                 [sigmoid] -> p
  2. loss     -- p and y flow into [loss] -> L
  3. backward -- gradients flow right to left through the same three boxes;
                 each weight keeps showing its current value, with the
                 gradient drawn as a second line just below it
  4. update   -- the weight labels feeding the linear box flip in place to
                 their new value (no "old -> new" clutter -- the color
                 change is what signals the update happened)

For clarity the diagram tracks a single hand-picked example (x = (2, 1),
y = 1) with simple, explicit starting weights (w = (-0.5, 0.3), b = 0,
lr = 0.5) -- not numbers pulled from an RNG-generated dataset -- trained
with the *same* forward/bce_loss/backward formulas as the notebook's
train() function, just with n=1, i.e. plain (non-batched) gradient descent
on one point, not the real 120-point full-batch training shown elsewhere in
the lesson. Every number on screen can be reproduced by hand (or in a
calculator) directly from those starting values and the formulas in the
lesson text -- that's the point of keeping them simple rather than
realistic. A few checkpoints along that single-example trajectory (starts
out misclassified, ends up converged/confident) are each played through all
4 stages, labeled with a short descriptive phase ("just initialized" ..
"converged") rather than a raw step count, since this toy trajectory isn't
meant to be compared against the real training's epoch numbers.

Lesson 32's make_mlp_training_loop_demo.py reuses this same box/arrow
vocabulary (stage colors, timing, flow dots, "operation boxes + variable
labels" layout) extended to a second [linear] -> [ReLU] stage, so the two
animations read as a matched pair.

Run from anywhere:
    python make_training_loop_demo.py
Output:
    training_loop_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "training_loop_demo.gif")

FORWARD_COLOR = "#e07b00"
LOSS_COLOR = "#d62728"
BACKWARD_COLOR = "#1f6fb4"
UPDATE_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#999999"
TEXT_COLOR = "#222222"

ACTIVE_LW = 3.4
IDLE_LW = 1.2
ACTIVE_BOX_LW = 3.2
IDLE_BOX_LW = 1.4

# A hand-picked, simple example -- not drawn from the notebook's randomly
# generated dataset -- specifically so a reader can redo every number in
# this animation themselves with a calculator: pick these same x, y, w, b,
# apply the forward/bce_loss/backward formulas below, and the numbers
# should match the diagram exactly at every checkpoint.
x_ex = np.array([2.0, 1.0])
y_ex = 1.0


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(x, w, b):
    z = x @ w + b
    return sigmoid(z), z


def bce_loss(p, y, eps=1e-9):
    return -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


def backward(x, p, y):
    grad_z = p - y
    grad_w = x * grad_z
    grad_b = grad_z
    return grad_w, grad_b


LR = 0.5
N_STEPS = 4
w = np.array([-0.5, 0.3])
b = 0.0

# Consecutive steps -- no gap between checkpoints, so each one's "update"
# result is exactly the next one's starting point. Nothing is skipped, so
# there's nothing that needs a fast-forward/transition to explain.
CHECKPOINT_STEPS = [0, 1, 2, 3]
PHASE_LABEL = {
    0: "before any updates",
    1: "after 1 update",
    2: "after 2 updates",
    3: "after 3 updates",
}
checkpoints = []
for step in range(N_STEPS):
    p, z = forward(x_ex, w, b)
    loss = bce_loss(p, y_ex)
    grad_w, grad_b = backward(x_ex, p, y_ex)
    if step in CHECKPOINT_STEPS:
        checkpoints.append(dict(
            step=step, w=w.copy(), b=b, z=z, p=p, loss=loss,
            grad_w=grad_w.copy(), grad_b=grad_b,
            w_new=w - LR * grad_w, b_new=b - LR * grad_b,
        ))
    w -= LR * grad_w
    b -= LR * grad_b

# ---- Diagram layout --------------------------------------------------------
# Inputs are plain text anchors (no shape); boxes are the three operations.
X1_POS, X2_POS, BIAS_POS = (0.0, 1.5), (0.0, -1.5), (0.0, 0.0)
Y_POS = (8.3, -2.0)
LIN_C, SIG_C, LOSS_C = (2.9, 0.0), (5.4, 0.0), (8.3, 0.0)
BOX_W, BOX_H = 1.5, 1.3

STAGES = ["forward", "loss", "backward", "update"]
STAGE_TITLE = {
    "forward": "1. forward pass",
    "loss": "2. loss",
    "backward": "3. backward pass",
    "update": "4. update",
}
STAGE_COLOR = {
    "forward": FORWARD_COLOR, "loss": LOSS_COLOR,
    "backward": BACKWARD_COLOR, "update": UPDATE_COLOR,
}


def add_box(ax, center, label, color, dashed=False):
    x, y_ = center
    lw = IDLE_BOX_LW if color == NEUTRAL_COLOR else ACTIVE_BOX_LW
    box = FancyBboxPatch((x - BOX_W / 2, y_ - BOX_H / 2), BOX_W, BOX_H,
                          boxstyle="round,pad=0.02,rounding_size=0.12",
                          facecolor="white", edgecolor=color, linewidth=lw,
                          linestyle="dashed" if dashed else "solid", zorder=3)
    ax.add_patch(box)
    ax.text(x, y_, label, ha="center", va="center", fontsize=11,
             fontweight="bold", color=TEXT_COLOR, zorder=4)
    return box


def add_var_anchor(ax, pos, text, side="above"):
    """A plain-text variable (no shape) serving as an arrow endpoint,
    e.g. 'x1 = 3.22' as a single combined label."""
    dy = 0.3 if side == "above" else -0.3
    ax.text(pos[0], pos[1] + dy, text, ha="center",
             va="bottom" if side == "above" else "top", fontsize=9.5,
             fontweight="bold", color=TEXT_COLOR, zorder=4)


def add_arrow(ax, pos_a, pos_b, color, active, patchA=None, patchB=None, reverse=False):
    lw = ACTIVE_LW if active else IDLE_LW
    a, b_ = (pos_b, pos_a) if reverse else (pos_a, pos_b)
    pA, pB = (patchB, patchA) if reverse else (patchA, patchB)
    arrow = FancyArrowPatch(a, b_, patchA=pA, patchB=pB,
                             shrinkA=10 if pA is None else 0,
                             shrinkB=10 if pB is None else 0,
                             arrowstyle="-|>", mutation_scale=18 if active else 14,
                             linewidth=lw, color=color, zorder=2)
    ax.add_patch(arrow)
    return a, b_


def flow_dots(ax, a, b_, color, t, n=3):
    """n small dots marching from a toward b_, looping with phase t in [0,1)."""
    for k in range(n):
        frac = 0.14 + ((t + k / n) % 1.0) * 0.72
        x = a[0] + (b_[0] - a[0]) * frac
        y = a[1] + (b_[1] - a[1]) * frac
        ax.plot(x, y, marker="o", markersize=4.8, color=color, zorder=6,
                 markeredgewidth=0)


def param_label(ax, pos_a, pos_b, text, color, dy=0.3, t=0.42, sub_text=None, sub_color=None):
    mx = pos_a[0] + (pos_b[0] - pos_a[0]) * t
    my = pos_a[1] + (pos_b[1] - pos_a[1]) * t
    bbox = dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.9)
    ax.text(mx, my + dy, text, ha="center", va="center", fontsize=8,
             color=color, zorder=5, bbox=bbox)
    if sub_text is not None:
        sign = 1 if dy >= 0 else -1
        ax.text(mx, my + dy + sign * 0.27, sub_text, ha="center", va="center",
                 fontsize=7.5, color=sub_color, zorder=5, bbox=bbox)


def render(ckpt, stage, t, progress):
    w_, b_, z_, p_, loss_ = ckpt["w"], ckpt["b"], ckpt["z"], ckpt["p"], ckpt["loss"]
    gw, gb = ckpt["grad_w"], ckpt["grad_b"]
    w_new, b_new = ckpt["w_new"], ckpt["b_new"]
    sc = STAGE_COLOR[stage]

    fig, ax = plt.subplots(figsize=(9.6, 4.4), dpi=120)

    in_active = stage in ("forward", "backward", "update")
    mid_active = stage in ("forward", "backward")
    out_active = stage in ("loss", "backward")
    loss_in_active = stage == "loss"

    in_color = {"forward": FORWARD_COLOR, "backward": BACKWARD_COLOR,
                "update": UPDATE_COLOR}.get(stage, NEUTRAL_COLOR)
    mid_color = {"forward": FORWARD_COLOR, "backward": BACKWARD_COLOR}.get(stage, NEUTRAL_COLOR)
    out_color = {"loss": LOSS_COLOR, "backward": BACKWARD_COLOR}.get(stage, NEUTRAL_COLOR)
    y_color = LOSS_COLOR if stage == "loss" else NEUTRAL_COLOR

    lin_box = add_box(ax, LIN_C, r"$w^T x + b$", in_color if in_active or mid_active else NEUTRAL_COLOR)
    sig_box = add_box(ax, SIG_C, "sigmoid", mid_color if mid_active else NEUTRAL_COLOR)
    loss_box = add_box(ax, LOSS_C, "loss", out_color if (out_active or loss_in_active) else NEUTRAL_COLOR)

    reverse = stage == "backward"
    seg_x1 = add_arrow(ax, X1_POS, LIN_C, in_color, in_active, patchB=lin_box, reverse=reverse)
    seg_bias = add_arrow(ax, BIAS_POS, LIN_C, in_color, in_active, patchB=lin_box, reverse=reverse)
    seg_x2 = add_arrow(ax, X2_POS, LIN_C, in_color, in_active, patchB=lin_box, reverse=reverse)
    seg_mid = add_arrow(ax, LIN_C, SIG_C, mid_color, mid_active, patchA=lin_box, patchB=sig_box, reverse=reverse)
    seg_out = add_arrow(ax, SIG_C, LOSS_C, out_color, out_active, patchA=sig_box, patchB=loss_box, reverse=reverse)
    seg_y = add_arrow(ax, Y_POS, LOSS_C, y_color, loss_in_active, patchB=loss_box)

    # Moving flow dots along every currently-active arrow (forward/loss/
    # backward only -- "update" is an instantaneous nudge, not a flow).
    if stage != "update":
        if in_active:
            flow_dots(ax, *seg_x1, in_color, t)
            flow_dots(ax, *seg_bias, in_color, t)
            flow_dots(ax, *seg_x2, in_color, t)
        if mid_active:
            flow_dots(ax, *seg_mid, mid_color, t)
        if out_active:
            flow_dots(ax, *seg_out, out_color, t)
        if loss_in_active:
            flow_dots(ax, *seg_y, y_color, t)

    # Input variable labels (plain text, no shape), name and value combined.
    add_var_anchor(ax, X1_POS, f"x1 = {x_ex[0]:.2f}", side="above")
    add_var_anchor(ax, X2_POS, f"x2 = {x_ex[1]:.2f}", side="below")
    ax.text(BIAS_POS[0], BIAS_POS[1] + 0.3, "1", ha="center", va="bottom",
             fontsize=10, fontweight="bold", color=TEXT_COLOR, zorder=4)
    add_var_anchor(ax, Y_POS, f"y = {y_ex:.0f}", side="below")

    # z, p, L value labels, sitting just outside their producing box.
    ax.text(LIN_C[0] + BOX_W / 2 + 0.55, LIN_C[1] + 0.55, f"z={z_:.2f}",
             ha="center", fontsize=8.5, color=TEXT_COLOR, zorder=5)
    ax.text(SIG_C[0] + BOX_W / 2 + 0.55, SIG_C[1] + 0.55, f"p={p_:.2f}",
             ha="center", fontsize=8.5, color=TEXT_COLOR, zorder=5)
    ax.text(LOSS_C[0], LOSS_C[1] + BOX_H / 2 + 0.3, f"L={loss_:.2f}",
             ha="center", fontsize=8.5, color=TEXT_COLOR, zorder=5)

    # Parameter labels: the weight's current value always shown in neutral
    # text; backward additionally shows the gradient just below/above it;
    # update swaps the displayed value in place (no "old -> new" clutter --
    # the green color is what signals a change just happened).
    if stage == "backward":
        param_label(ax, X1_POS, LIN_C, f"w1={w_[0]:.2f}", TEXT_COLOR, dy=0.3,
                    sub_text=f"dL/dw1={gw[0]:+.2f}", sub_color=BACKWARD_COLOR)
        param_label(ax, X2_POS, LIN_C, f"w2={w_[1]:.2f}", TEXT_COLOR, dy=-0.3,
                    sub_text=f"dL/dw2={gw[1]:+.2f}", sub_color=BACKWARD_COLOR)
        param_label(ax, BIAS_POS, LIN_C, f"b={b_:.2f}", TEXT_COLOR, dy=0.28,
                    sub_text=f"dL/db={gb:+.2f}", sub_color=BACKWARD_COLOR)
    elif stage == "update":
        param_label(ax, X1_POS, LIN_C, f"w1={w_new[0]:.2f}", UPDATE_COLOR, dy=0.3)
        param_label(ax, X2_POS, LIN_C, f"w2={w_new[1]:.2f}", UPDATE_COLOR, dy=-0.3)
        param_label(ax, BIAS_POS, LIN_C, f"b={b_new:.2f}", UPDATE_COLOR, dy=0.28)
    else:
        param_label(ax, X1_POS, LIN_C, f"w1={w_[0]:.2f}", TEXT_COLOR, dy=0.3)
        param_label(ax, X2_POS, LIN_C, f"w2={w_[1]:.2f}", TEXT_COLOR, dy=-0.3)
        param_label(ax, BIAS_POS, LIN_C, f"b={b_:.2f}", TEXT_COLOR, dy=0.28)

    ax.set_xlim(-0.9, 10.1)
    ax.set_ylim(-3.0, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(0.5, 1.14, STAGE_TITLE[stage], transform=ax.transAxes, ha="center",
             fontsize=12.5, color=sc, fontweight="bold")

    add_progress_bar(ax, progress, sc)
    fig.subplots_adjust(top=0.82, bottom=0.05)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def add_progress_bar(ax, progress, color):
    """Training-progress bar in place of a wordy status line: fill advances
    smoothly across the whole animation, colored to match whatever's active."""
    bar_x0, bar_x1, bar_y, bar_h = 0.32, 0.68, 1.045, 0.032
    ax.add_patch(Rectangle((bar_x0, bar_y), bar_x1 - bar_x0, bar_h,
                             transform=ax.transAxes, facecolor="#e5e5e5",
                             edgecolor="none", zorder=4, clip_on=False))
    ax.add_patch(Rectangle((bar_x0, bar_y), (bar_x1 - bar_x0) * progress, bar_h,
                             transform=ax.transAxes, facecolor=color,
                             edgecolor="none", zorder=5, clip_on=False))


frames = []
HOLD = 50
HOLD_LAST = 80
CYCLES_PER_HOLD = 2.2
n_units = len(checkpoints) * len(STAGES)
for i, ckpt in enumerate(checkpoints):
    for j, stage in enumerate(STAGES):
        is_last = (i == len(checkpoints) - 1) and (j == len(STAGES) - 1)
        n = HOLD_LAST if is_last else HOLD
        progress = (i * len(STAGES) + j) / (n_units - 1)
        for k in range(n):
            t = (k / HOLD) * CYCLES_PER_HOLD % 1.0
            frames.append(render(ckpt, stage, t, progress))

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=60,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
