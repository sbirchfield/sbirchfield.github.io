"""Generate mlp_training_loop_demo.gif: the lesson 32 companion to lesson 31's
training_loop_demo.gif, extending the same computation-graph vocabulary to a
2-input -> 2-hidden-ReLU-unit -> 1-output-sigmoid network. Same idea as
before -- boxes are operations, variables are plain text labels riding the
arrows between them, and the diagram cycles through 4 highlighted stages --
but now the forward/backward paths each have two hops instead of one: a
linear box and a ReLU box per hidden unit (drawn as two parallel rows), then
the second linear box, sigmoid, and loss exactly as in lesson 31.

  1. forward  -- inputs -> [w0.x+b0]->[ReLU] and [w1.x+b1]->[ReLU] (parallel)
                 -> both hidden outputs -> [w2.h+b2] -> [sigmoid] -> p
  2. loss     -- p and y -> [loss] -> L
  3. backward -- gradients flow right to left through all five boxes
  4. update   -- every weight label flips in place to its new value

As in lesson 31's animation, the example here is a small hand-picked
example -- not a randomly generated dataset -- specifically so a reader can
redo every number themselves: x = (2, 1), y = 1, with simple explicit
starting weights for both hidden units and the output layer (see below).
The 4 checkpoints are consecutive steps (0, 1, 2, 3) of plain (non-batched)
gradient descent using the *same* forward/backward formulas as the
notebook's 2-layer MLP, so there is no gap between checkpoints and nothing
needs to be fast-forwarded over.

Run from anywhere:
    python make_mlp_training_loop_demo.py
Output:
    mlp_training_loop_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "mlp_training_loop_demo.gif")

FORWARD_COLOR = "#e07b00"
LOSS_COLOR = "#d62728"
BACKWARD_COLOR = "#1f6fb4"
UPDATE_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#999999"
TEXT_COLOR = "#222222"

ACTIVE_LW = 3.0
IDLE_LW = 1.1
ACTIVE_BOX_LW = 2.8
IDLE_BOX_LW = 1.3

# A hand-picked, simple example -- same x, y as lesson 31's animation, now
# run through a 2-hidden-unit network -- so a reader can redo every number
# here with a calculator: pick these same x, y, W1, b1, W2, b2, apply the
# forward/bce_loss/backward formulas below, and the numbers should match
# the diagram exactly at every checkpoint.
x_ex = np.array([2.0, 1.0])
y_ex = 1.0


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def relu_deriv(z):
    return (z > 0).astype(np.float64)


def forward(x, W1, b1, W2, b2):
    z1 = x @ W1 + b1          # shape (2,): one pre-activation per hidden unit
    a1 = relu(z1)
    z2 = a1 @ W2 + b2         # scalar
    p = sigmoid(z2)
    return p, (z1, a1, z2)


def bce_loss(p, y, eps=1e-9):
    return -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


def backward(x, y, p, cache, W2):
    z1, a1, z2 = cache
    grad_z2 = p - y
    grad_W2 = a1 * grad_z2
    grad_b2 = grad_z2
    grad_a1 = grad_z2 * W2
    grad_z1 = grad_a1 * relu_deriv(z1)
    grad_W1 = np.outer(x, grad_z1)   # shape (2, 2): grad_W1[i, j] = x[i] * grad_z1[j]
    grad_b1 = grad_z1
    return grad_W1, grad_b1, grad_W2, grad_b2


LR = 0.5
N_STEPS = 4
# W1[:, j] is hidden unit j's weight vector; b1[j] its bias.
W1 = np.array([[0.2, -0.3],
               [-0.4, 0.5]])
b1 = np.array([0.1, 0.3])
W2 = np.array([0.4, 0.5])
b2 = -0.6

CHECKPOINT_STEPS = [0, 1, 2, 3]
PHASE_LABEL = {
    0: "before any updates",
    1: "after 1 update",
    2: "after 2 updates",
    3: "after 3 updates",
}
checkpoints = []
for step in range(N_STEPS):
    p, cache = forward(x_ex, W1, b1, W2, b2)
    z1, a1, z2 = cache
    loss = bce_loss(p, y_ex)
    grad_W1, grad_b1, grad_W2, grad_b2 = backward(x_ex, y_ex, p, cache, W2)
    if step in CHECKPOINT_STEPS:
        checkpoints.append(dict(
            step=step, W1=W1.copy(), b1=b1.copy(), W2=W2.copy(), b2=b2,
            z1=z1.copy(), a1=a1.copy(), z2=z2, p=p, loss=loss,
            grad_W1=grad_W1.copy(), grad_b1=grad_b1.copy(),
            grad_W2=grad_W2.copy(), grad_b2=grad_b2,
            W1_new=W1 - LR * grad_W1, b1_new=b1 - LR * grad_b1,
            W2_new=W2 - LR * grad_W2, b2_new=b2 - LR * grad_b2,
        ))
    W1 = W1 - LR * grad_W1
    b1 = b1 - LR * grad_b1
    W2 = W2 - LR * grad_W2
    b2 = b2 - LR * grad_b2

# ---- Diagram layout --------------------------------------------------------
X1_POS, X2_POS, BIAS1_POS = (0.0, 3.3), (0.0, -3.3), (0.0, 0.0)
LIN0_C, RELU0_C = (3.0, 2.1), (5.2, 2.1)
LIN1_C, RELU1_C = (3.0, -2.1), (5.2, -2.1)
BIAS2_POS = (5.2, 0.0)
OUT_C, SIG_C, LOSS_C = (7.6, 0.0), (9.8, 0.0), (12.0, 0.0)
Y_POS = (12.0, -2.4)
BOX_W, BOX_H = 1.5, 1.1

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


def add_box(ax, center, label, color, dashed=False, w=BOX_W, h=BOX_H, fontsize=11):
    x, y_ = center
    lw = IDLE_BOX_LW if color == NEUTRAL_COLOR else ACTIVE_BOX_LW
    box = FancyBboxPatch((x - w / 2, y_ - h / 2), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor="white", edgecolor=color, linewidth=lw,
                          linestyle="dashed" if dashed else "solid", zorder=3)
    ax.add_patch(box)
    ax.text(x, y_, label, ha="center", va="center", fontsize=fontsize,
             fontweight="bold", color=TEXT_COLOR, zorder=4)
    return box


def add_var_anchor(ax, pos, text, side="above"):
    dy = 0.3 if side == "above" else -0.3
    ax.text(pos[0], pos[1] + dy, text, ha="center",
             va="bottom" if side == "above" else "top", fontsize=9.5,
             fontweight="bold", color=TEXT_COLOR, zorder=4)


def add_arrow(ax, pos_a, pos_b, color, active, patchA=None, patchB=None, reverse=False):
    lw = ACTIVE_LW if active else IDLE_LW
    a, b_ = (pos_b, pos_a) if reverse else (pos_a, pos_b)
    pA, pB = (patchB, patchA) if reverse else (patchA, patchB)
    arrow = FancyArrowPatch(a, b_, patchA=pA, patchB=pB,
                             shrinkA=9 if pA is None else 0,
                             shrinkB=9 if pB is None else 0,
                             arrowstyle="-|>", mutation_scale=16 if active else 12,
                             linewidth=lw, color=color, zorder=2)
    ax.add_patch(arrow)
    return a, b_


def flow_dots(ax, a, b_, color, t, n=3):
    for k in range(n):
        frac = 0.14 + ((t + k / n) % 1.0) * 0.72
        x = a[0] + (b_[0] - a[0]) * frac
        y = a[1] + (b_[1] - a[1]) * frac
        ax.plot(x, y, marker="o", markersize=4.4, color=color, zorder=6,
                 markeredgewidth=0)


def param_label(ax, pos_a, pos_b, text, color, dy=0.3, t=0.2, sub_text=None, sub_color=None, fontsize=7.5):
    mx = pos_a[0] + (pos_b[0] - pos_a[0]) * t
    my = pos_a[1] + (pos_b[1] - pos_a[1]) * t
    bbox = dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.9)
    ax.text(mx, my + dy, text, ha="center", va="center", fontsize=fontsize,
             color=color, zorder=5, bbox=bbox)
    if sub_text is not None:
        sign = 1 if dy >= 0 else -1
        ax.text(mx, my + dy + sign * 0.17, sub_text, ha="center", va="center",
                 fontsize=fontsize - 0.5, color=sub_color, zorder=5, bbox=bbox)


def add_progress_bar(ax, progress, color):
    bar_x0, bar_x1, bar_y, bar_h = 0.34, 0.66, 1.05, 0.03
    ax.add_patch(Rectangle((bar_x0, bar_y), bar_x1 - bar_x0, bar_h,
                             transform=ax.transAxes, facecolor="#e5e5e5",
                             edgecolor="none", zorder=4, clip_on=False))
    ax.add_patch(Rectangle((bar_x0, bar_y), (bar_x1 - bar_x0) * progress, bar_h,
                             transform=ax.transAxes, facecolor=color,
                             edgecolor="none", zorder=5, clip_on=False))


def render(ckpt, stage, t, progress):
    W1_, b1_, W2_, b2_ = ckpt["W1"], ckpt["b1"], ckpt["W2"], ckpt["b2"]
    z1_, a1_, z2_, p_, loss_ = ckpt["z1"], ckpt["a1"], ckpt["z2"], ckpt["p"], ckpt["loss"]
    gW1, gb1, gW2, gb2 = ckpt["grad_W1"], ckpt["grad_b1"], ckpt["grad_W2"], ckpt["grad_b2"]
    W1n, b1n, W2n, b2n = ckpt["W1_new"], ckpt["b1_new"], ckpt["W2_new"], ckpt["b2_new"]
    sc = STAGE_COLOR[stage]

    fig, ax = plt.subplots(figsize=(12.6, 6.4), dpi=112)

    in_active = stage in ("forward", "backward", "update")
    hid_active = stage in ("forward", "backward")
    out_active = stage in ("forward", "backward")
    loss_in_active = stage in ("loss", "backward")
    y_active = stage == "loss"

    in_color = {"forward": FORWARD_COLOR, "backward": BACKWARD_COLOR,
                "update": UPDATE_COLOR}.get(stage, NEUTRAL_COLOR)
    hid_color = {"forward": FORWARD_COLOR, "backward": BACKWARD_COLOR}.get(stage, NEUTRAL_COLOR)
    out_color = {"forward": FORWARD_COLOR, "backward": BACKWARD_COLOR}.get(stage, NEUTRAL_COLOR)
    loss_color = {"loss": LOSS_COLOR, "backward": BACKWARD_COLOR}.get(stage, NEUTRAL_COLOR)
    y_color = LOSS_COLOR if y_active else NEUTRAL_COLOR

    lin0_box = add_box(ax, LIN0_C, r"$w_0^Tx+b_0$", in_color if (in_active or hid_active) else NEUTRAL_COLOR, fontsize=9)
    relu0_box = add_box(ax, RELU0_C, "ReLU", hid_color if hid_active else NEUTRAL_COLOR)
    lin1_box = add_box(ax, LIN1_C, r"$w_1^Tx+b_1$", in_color if (in_active or hid_active) else NEUTRAL_COLOR, fontsize=9)
    relu1_box = add_box(ax, RELU1_C, "ReLU", hid_color if hid_active else NEUTRAL_COLOR)
    out_box = add_box(ax, OUT_C, r"$w_2^Th+b_2$", out_color if (hid_active or out_active) else NEUTRAL_COLOR, fontsize=9)
    sig_box = add_box(ax, SIG_C, "sigmoid", out_color if out_active else NEUTRAL_COLOR)
    loss_box = add_box(ax, LOSS_C, "loss", loss_color if (loss_in_active) else NEUTRAL_COLOR)

    reverse = stage == "backward"
    seg_x1_0 = add_arrow(ax, X1_POS, LIN0_C, in_color, in_active, patchB=lin0_box, reverse=reverse)
    seg_x2_0 = add_arrow(ax, X2_POS, LIN0_C, in_color, in_active, patchB=lin0_box, reverse=reverse)
    seg_b1_0 = add_arrow(ax, BIAS1_POS, LIN0_C, in_color, in_active, patchB=lin0_box, reverse=reverse)
    seg_x1_1 = add_arrow(ax, X1_POS, LIN1_C, in_color, in_active, patchB=lin1_box, reverse=reverse)
    seg_x2_1 = add_arrow(ax, X2_POS, LIN1_C, in_color, in_active, patchB=lin1_box, reverse=reverse)
    seg_b1_1 = add_arrow(ax, BIAS1_POS, LIN1_C, in_color, in_active, patchB=lin1_box, reverse=reverse)

    seg_relu0 = add_arrow(ax, LIN0_C, RELU0_C, hid_color, hid_active, patchA=lin0_box, patchB=relu0_box, reverse=reverse)
    seg_relu1 = add_arrow(ax, LIN1_C, RELU1_C, hid_color, hid_active, patchA=lin1_box, patchB=relu1_box, reverse=reverse)

    seg_h0 = add_arrow(ax, RELU0_C, OUT_C, out_color, out_active, patchA=relu0_box, patchB=out_box, reverse=reverse)
    seg_h1 = add_arrow(ax, RELU1_C, OUT_C, out_color, out_active, patchA=relu1_box, patchB=out_box, reverse=reverse)
    seg_b2 = add_arrow(ax, BIAS2_POS, OUT_C, out_color, out_active, patchB=out_box, reverse=reverse)

    seg_sig = add_arrow(ax, OUT_C, SIG_C, out_color, out_active, patchA=out_box, patchB=sig_box, reverse=reverse)
    seg_loss = add_arrow(ax, SIG_C, LOSS_C, loss_color, loss_in_active, patchA=sig_box, patchB=loss_box, reverse=reverse)
    seg_y = add_arrow(ax, Y_POS, LOSS_C, y_color, y_active, patchB=loss_box)

    if stage != "update":
        for seg, color, active in [
            (seg_x1_0, in_color, in_active), (seg_x2_0, in_color, in_active), (seg_b1_0, in_color, in_active),
            (seg_x1_1, in_color, in_active), (seg_x2_1, in_color, in_active), (seg_b1_1, in_color, in_active),
            (seg_relu0, hid_color, hid_active), (seg_relu1, hid_color, hid_active),
            (seg_h0, out_color, out_active), (seg_h1, out_color, out_active), (seg_b2, out_color, out_active),
            (seg_sig, out_color, out_active), (seg_loss, loss_color, loss_in_active),
            (seg_y, y_color, y_active),
        ]:
            if active:
                flow_dots(ax, *seg, color, t)

    add_var_anchor(ax, X1_POS, f"x1 = {x_ex[0]:.2f}", side="above")
    add_var_anchor(ax, X2_POS, f"x2 = {x_ex[1]:.2f}", side="below")
    ax.text(BIAS1_POS[0], BIAS1_POS[1] + 0.26, "1", ha="center", va="bottom",
             fontsize=10, fontweight="bold", color=TEXT_COLOR, zorder=4)
    ax.text(BIAS2_POS[0], BIAS2_POS[1] + 0.26, "1", ha="center", va="bottom",
             fontsize=10, fontweight="bold", color=TEXT_COLOR, zorder=4)
    add_var_anchor(ax, Y_POS, f"y = {y_ex:.0f}", side="below")

    ax.text(RELU0_C[0] + BOX_W / 2 + 0.48, RELU0_C[1] + 0.5, f"h0={a1_[0]:.2f}", ha="center", fontsize=7.8, color=TEXT_COLOR, zorder=5)
    ax.text(RELU1_C[0] + BOX_W / 2 + 0.48, RELU1_C[1] - 0.5, f"h1={a1_[1]:.2f}", ha="center", fontsize=7.8, color=TEXT_COLOR, zorder=5)
    ax.text(OUT_C[0] + BOX_W / 2 + 0.48, OUT_C[1] + 0.5, f"z2={z2_:.2f}", ha="center", fontsize=7.8, color=TEXT_COLOR, zorder=5)
    ax.text(SIG_C[0] + BOX_W / 2 + 0.48, SIG_C[1] + 0.5, f"p={p_:.2f}", ha="center", fontsize=7.8, color=TEXT_COLOR, zorder=5)
    ax.text(LOSS_C[0], LOSS_C[1] + BOX_H / 2 + 0.28, f"L={loss_:.2f}", ha="center", fontsize=7.8, color=TEXT_COLOR, zorder=5)

    if stage == "backward":
        param_label(ax, X1_POS, LIN0_C, f"w0_1={W1_[0,0]:.2f}", TEXT_COLOR, dy=0.2, t=0.3, fontsize=6.5,
                    sub_text=f"dL/dw0_1={gW1[0,0]:+.2f}", sub_color=BACKWARD_COLOR)
        param_label(ax, X2_POS, LIN0_C, f"w0_2={W1_[1,0]:.2f}", TEXT_COLOR, dy=-0.2, t=0.3, fontsize=6.5,
                    sub_text=f"dL/dw0_2={gW1[1,0]:+.2f}", sub_color=BACKWARD_COLOR)
        param_label(ax, BIAS1_POS, LIN0_C, f"b0={b1_[0]:.2f}", TEXT_COLOR, dy=0.22, t=0.2, fontsize=6.5,
                    sub_text=f"dL/db0={gb1[0]:+.2f}", sub_color=BACKWARD_COLOR)
        param_label(ax, X1_POS, LIN1_C, f"w1_1={W1_[0,1]:.2f}", TEXT_COLOR, dy=-0.2, t=0.3, fontsize=6.5,
                    sub_text=f"dL/dw1_1={gW1[0,1]:+.2f}", sub_color=BACKWARD_COLOR)
        param_label(ax, X2_POS, LIN1_C, f"w1_2={W1_[1,1]:.2f}", TEXT_COLOR, dy=0.2, t=0.3, fontsize=6.5,
                    sub_text=f"dL/dw1_2={gW1[1,1]:+.2f}", sub_color=BACKWARD_COLOR)
        param_label(ax, BIAS1_POS, LIN1_C, f"b1={b1_[1]:.2f}", TEXT_COLOR, dy=-0.22, t=0.2, fontsize=6.5,
                    sub_text=f"dL/db1={gb1[1]:+.2f}", sub_color=BACKWARD_COLOR)
        param_label(ax, RELU0_C, OUT_C, f"w2_1={W2_[0]:.2f}", TEXT_COLOR, dy=0.26, t=0.45,
                    sub_text=f"dL/dw2_1={gW2[0]:+.2f}", sub_color=BACKWARD_COLOR)
        param_label(ax, RELU1_C, OUT_C, f"w2_2={W2_[1]:.2f}", TEXT_COLOR, dy=-0.26, t=0.45,
                    sub_text=f"dL/dw2_2={gW2[1]:+.2f}", sub_color=BACKWARD_COLOR)
        param_label(ax, BIAS2_POS, OUT_C, f"b2={b2_:.2f}", TEXT_COLOR, dy=0.22, t=0.2,
                    sub_text=f"dL/db2={gb2:+.2f}", sub_color=BACKWARD_COLOR)
    elif stage == "update":
        param_label(ax, X1_POS, LIN0_C, f"w0_1={W1n[0,0]:.2f}", UPDATE_COLOR, dy=0.2, t=0.3, fontsize=6.5)
        param_label(ax, X2_POS, LIN0_C, f"w0_2={W1n[1,0]:.2f}", UPDATE_COLOR, dy=-0.2, t=0.3, fontsize=6.5)
        param_label(ax, BIAS1_POS, LIN0_C, f"b0={b1n[0]:.2f}", UPDATE_COLOR, dy=0.22, t=0.2, fontsize=6.5)
        param_label(ax, X1_POS, LIN1_C, f"w1_1={W1n[0,1]:.2f}", UPDATE_COLOR, dy=-0.2, t=0.3, fontsize=6.5)
        param_label(ax, X2_POS, LIN1_C, f"w1_2={W1n[1,1]:.2f}", UPDATE_COLOR, dy=0.2, t=0.3, fontsize=6.5)
        param_label(ax, BIAS1_POS, LIN1_C, f"b1={b1n[1]:.2f}", UPDATE_COLOR, dy=-0.22, t=0.2, fontsize=6.5)
        param_label(ax, RELU0_C, OUT_C, f"w2_1={W2n[0]:.2f}", UPDATE_COLOR, dy=0.26, t=0.45)
        param_label(ax, RELU1_C, OUT_C, f"w2_2={W2n[1]:.2f}", UPDATE_COLOR, dy=-0.26, t=0.45)
        param_label(ax, BIAS2_POS, OUT_C, f"b2={b2n:.2f}", UPDATE_COLOR, dy=0.22, t=0.2)
    else:
        param_label(ax, X1_POS, LIN0_C, f"w0_1={W1_[0,0]:.2f}", TEXT_COLOR, dy=0.2, t=0.3, fontsize=6.5)
        param_label(ax, X2_POS, LIN0_C, f"w0_2={W1_[1,0]:.2f}", TEXT_COLOR, dy=-0.2, t=0.3, fontsize=6.5)
        param_label(ax, BIAS1_POS, LIN0_C, f"b0={b1_[0]:.2f}", TEXT_COLOR, dy=0.22, t=0.2, fontsize=6.5)
        param_label(ax, X1_POS, LIN1_C, f"w1_1={W1_[0,1]:.2f}", TEXT_COLOR, dy=-0.2, t=0.3, fontsize=6.5)
        param_label(ax, X2_POS, LIN1_C, f"w1_2={W1_[1,1]:.2f}", TEXT_COLOR, dy=0.2, t=0.3, fontsize=6.5)
        param_label(ax, BIAS1_POS, LIN1_C, f"b1={b1_[1]:.2f}", TEXT_COLOR, dy=-0.22, t=0.2, fontsize=6.5)
        param_label(ax, RELU0_C, OUT_C, f"w2_1={W2_[0]:.2f}", TEXT_COLOR, dy=0.26, t=0.45)
        param_label(ax, RELU1_C, OUT_C, f"w2_2={W2_[1]:.2f}", TEXT_COLOR, dy=-0.26, t=0.45)
        param_label(ax, BIAS2_POS, OUT_C, f"b2={b2_:.2f}", TEXT_COLOR, dy=0.22, t=0.2)

    ax.set_xlim(-0.9, 13.0)
    ax.set_ylim(-4.3, 4.1)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(0.5, 1.1, STAGE_TITLE[stage], transform=ax.transAxes, ha="center",
             fontsize=12.5, color=sc, fontweight="bold")
    add_progress_bar(ax, progress, sc)
    fig.subplots_adjust(top=0.84, bottom=0.04, left=0.03, right=0.97)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


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
