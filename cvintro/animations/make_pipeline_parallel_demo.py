"""Generate pipeline_parallel_demo.gif: an animated schematic companion to
the lesson 44 notebook's "Model parallelism: shipping activations across a
boundary" section. The point of this demo is narrow and specific: prove that
manually splitting a model across two "devices" (detach the activation at
the boundary, reattach the gradient on the way back) gives the exact same
gradient as running the whole model on one device. It is a correctness
check, not a speed demo -- each device is shown solid (active/computing) or
hollow (idle) at every beat, so the idle time is visible on purpose rather
than explained away. Real pipeline parallelism's actual throughput win comes
from overlapping many microbatches so devices rarely sit idle like this; the
notebook text right above this animation covers that "bubble" caveat in
words, so the animation itself stays to the one thing a picture does best:
is the split-and-reattach math the same as not splitting at all.

Reuses the notebook's own TwoStageNet / manual detach-reattach demo
verbatim (model seed=2, same X/y), so the reported gradient match is
exactly 0.0, as in the notebook.

Run from anywhere:
    python make_pipeline_parallel_demo.py
Output:
    pipeline_parallel_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "pipeline_parallel_demo.gif")

TEXT_COLOR = "#222222"
STAGE_A_COLOR = "#1f6fb4"
STAGE_B_COLOR = "#2e9e44"
ACTIVATION_COLOR = "#f2994a"
GRADIENT_COLOR = "#d62728"
IDLE_COLOR = "#999999"


class TwoStageNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.stage2 = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.stage2(self.stage1(x))


torch.manual_seed(2)
whole_model = TwoStageNet()
X = torch.randn(8, 8)
y = torch.randn(8, 1)

F.mse_loss(whole_model(X), y).backward()
grad_whole = [p.grad.clone() for p in whole_model.parameters()]

torch.manual_seed(2)
split_model = TwoStageNet()
split_model.zero_grad()
activation_out = split_model.stage1(X)
activation_in = activation_out.detach().requires_grad_(True)
prediction = split_model.stage2(activation_in)
F.mse_loss(prediction, y).backward()
activation_grad = activation_in.grad
activation_out.backward(activation_grad)

grad_split = [p.grad.clone() for p in split_model.parameters()]
max_diff = max((g_whole - g_split).abs().max().item() for g_whole, g_split in zip(grad_whole, grad_split))
print(f"max abs diff, whole model vs. split across 2 stages: {max_diff:.2e}")

A_X, A_Y, A_W, A_H = 0.6, 1.6, 2.8, 3.0
B_X, B_Y, B_W, B_H = 8.0, 1.6, 2.8, 3.0
BOUNDARY_X = (A_X + A_W + B_X) / 2
CENTER_X = (A_X + B_X + B_W) / 2
ARROW_Y = A_Y + A_H / 2

BEATS = ["intro", "fwd_A", "fwd_send", "bwd_B", "bwd_send", "final"]
HOLD = {"intro": 40, "fwd_A": 35, "fwd_send": 45, "bwd_B": 35, "bwd_send": 45, "final": 90}

# Short, single-line captions only -- the notebook text above the animation
# already covers the "why idle time is OK" and "real systems use
# microbatches" nuance in words, so the animation itself stays wordless
# except for naming what's on screen right now.
CAPTION = {
    "intro": "same model, split across 2 devices",
    "fwd_A": "device A computes stage 1",
    "fwd_send": "activation sent to device B",
    "bwd_B": "device B computes the loss, then its gradient",
    "bwd_send": "gradient sent back to device A",
    "final": "split gradient vs. unsplit gradient",
}

# Which device is solid/active (computing) at each beat -- everything else
# is hollow/idle. This is the whole visual argument: watch which box is
# filled in, no extra idle-time caption needed.
ACTIVE = {
    "intro": set(),
    "fwd_A": {"A"},
    "fwd_send": {"B"},
    "bwd_B": {"B"},
    "bwd_send": {"A"},
    "final": set(),
}

# (direction, color, label) of the one arrow shown this beat, or None.
ARROW = {
    "intro": None, "fwd_A": None,
    "fwd_send": ("fwd", ACTIVATION_COLOR, "activation"),
    "bwd_B": None,
    "bwd_send": ("bwd", GRADIENT_COLOR, "gradient"),
    "final": None,
}


def draw_box(ax, x, y, w, h, color, label, active):
    fill = color if active else "white"
    text_color = "white" if active else color
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                         edgecolor=color, facecolor=fill, linewidth=2.6, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11,
            color=text_color, fontweight="bold", zorder=4)
    if not active:
        ax.text(x + w / 2, y - 0.3, "idle", ha="center", va="top", fontsize=8.5,
                color=IDLE_COLOR, style="italic")


def render(beat):
    fig, ax = plt.subplots(figsize=(11.6, 5.6), dpi=120)
    fig.subplots_adjust(top=0.80, bottom=0.04, left=0.02, right=0.98)
    ax.set_xlim(0, 11.2)
    ax.set_ylim(-0.8, 5.4)
    ax.axis("off")

    active = ACTIVE[beat]
    draw_box(ax, A_X, A_Y, A_W, A_H, STAGE_A_COLOR, "device A\nstage 1", "A" in active)
    draw_box(ax, B_X, B_Y, B_W, B_H, STAGE_B_COLOR, "device B\nstage 2", "B" in active)

    ax.plot([BOUNDARY_X, BOUNDARY_X], [1.0, 4.9], color="#cccccc", linewidth=1.2, linestyle=":", zorder=1)

    arrow = ARROW[beat]
    if arrow is not None:
        direction, color, label = arrow
        x0, x1 = (A_X + A_W, B_X) if direction == "fwd" else (B_X, A_X + A_W)
        ax.annotate("", xy=(x1, ARROW_Y), xytext=(x0, ARROW_Y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.6), zorder=2)
        ax.text((x0 + x1) / 2, ARROW_Y + 0.3, label, ha="center", va="bottom",
                fontsize=10, color=color, fontweight="bold")

    if beat == "intro":
        ax.text(CENTER_X, A_Y - 0.55, "goal: check that splitting the model doesn't change the math",
                ha="center", fontsize=9.5, color=TEXT_COLOR, style="italic")

    if beat == "final":
        match = max_diff == 0.0
        cmp_color = "#2e9e44" if match else "#d62728"
        note = "exact match (0.0, not just roundoff)" if match else f"diff: {max_diff:.1e}"
        ax.text(CENTER_X, A_Y - 0.55, note, ha="center", fontsize=11, color=cmp_color, fontweight="bold")

    fig.suptitle("pipeline parallelism: same gradient, computed on 2 devices", fontsize=13,
                 fontweight="bold", color=TEXT_COLOR, y=0.97)
    ax.set_title(CAPTION[beat], fontsize=12, color=TEXT_COLOR, pad=14)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
for beat in BEATS:
    frames += [render(beat)] * HOLD[beat]

frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=55, loop=0)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
