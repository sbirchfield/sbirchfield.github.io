"""Generate loss_scaling_sweep_demo.gif: an animated companion to the lesson
44 notebook's "Mixed-precision training: the recipe" section. Sweeps the
loss-scaling constant smoothly from 2**0 up through 2**28, showing the
first-layer gradient's magnitude distribution (after scaling, before the
fp16 cast) shift up past fp16's representable floor, with a horizontal
slider tracking the current scale and a fixed tick at the common real-world
default of 65536 (2**16).

Reuses the notebook's own DeepSigmoidNet class and loss-scaling loop
verbatim (model seed=0, same X/y, same model instance reused across scales,
.zero_grad() before each backward), so the fraction-still-zero numbers at
scale=1, 65536, 2**20, 2**24, 2**28 match the notebook exactly.

Run from anywhere:
    python make_loss_scaling_sweep_demo.py
Output:
    loss_scaling_sweep_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "loss_scaling_sweep_demo.gif")

TEXT_COLOR = "#222222"
BAR_COLOR = "#1f6fb4"
ZERO_COLOR = "#d62728"
FLOOR_COLOR = "#888888"
SLIDER_COLOR = "#bbbbbb"
HANDLE_COLOR = "#1f6fb4"
TICK_COLOR = "#2e9e44"


class DeepSigmoidNet(nn.Module):
    def __init__(self, depth=12, width=16, in_dim=4):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Sigmoid()]
            d = width
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


torch.manual_seed(0)
model = DeepSigmoidNet(depth=12)
X = torch.randn(32, 4)
y = torch.randn(32, 1)

FP16_NORMAL_FLOOR = torch.finfo(torch.float16).tiny       # ~6.1e-5
FP16_SUBNORMAL_FLOOR = 2 ** -24                             # ~6e-8

CHECKPOINTS = [0.0, 16.0, 20.0, 24.0, 28.0]  # log2(scale): matches the notebook's 1, 65536, 2**20, 2**24, 2**28
DEFAULT_LOG2_SCALE = 16.0  # 65536

print(f"{'loss scale':>12} {'fraction of fp16 grads still zero':>34}")
for log2_scale in CHECKPOINTS:
    scale = 2 ** log2_scale
    model.zero_grad()
    loss = F.mse_loss(model(X), y)
    (loss * scale).backward()
    grad_fp16 = model.net[0].weight.grad.half()
    print(f"{scale:>12.0f} {float((grad_fp16 == 0).float().mean()):>33.1%}")


def compute_frac_zero(log2_scale):
    scale = 2 ** log2_scale
    model.zero_grad()
    loss = F.mse_loss(model(X), y)
    (loss * scale).backward()
    grad = model.net[0].weight.grad
    grad_fp16 = grad.half()
    frac_zero = float((grad_fp16 == 0).float().mean())
    mags = grad.abs().flatten().numpy()
    return frac_zero, mags


SLIDER_MIN, SLIDER_MAX = 0.0, 28.0
TICKS = [(DEFAULT_LOG2_SCALE, "common default ($2^{16}$)")]

Y_MIN, Y_MAX = -14.0, 2.0  # log10 magnitude range


SLIDER_PAD = 1.0  # keeps the handle marker from being clipped at the slider's extremes


def draw_slider(fig, log2_scale):
    sax = fig.add_axes([0.14, 0.015, 0.72, 0.1])
    sax.set_xlim(SLIDER_MIN - SLIDER_PAD, SLIDER_MAX + SLIDER_PAD)
    sax.set_ylim(-1.1, 2.4)  # extra headroom above the "loss scale" text row
    sax.axis("off")

    sax.text(SLIDER_MIN, 1.3, "$2^0$", ha="left", va="bottom", fontsize=8, color=TEXT_COLOR)
    sax.text((SLIDER_MIN + SLIDER_MAX) / 2, 1.3, "loss scale (log2)", ha="center", va="bottom",
              fontsize=8.5, color=TEXT_COLOR, fontweight="bold")
    sax.text(SLIDER_MAX, 1.3, "$2^{28}$", ha="right", va="bottom", fontsize=8, color=TEXT_COLOR)

    sax.plot([SLIDER_MIN, SLIDER_MAX], [0.5, 0.5], color=SLIDER_COLOR, linewidth=5,
              solid_capstyle="round", zorder=1)

    for pos, label in TICKS:
        sax.plot([pos, pos], [-0.1, 1.1], color=TICK_COLOR, linewidth=2.2, zorder=2)
        sax.text(pos, -0.4, label, ha="center", va="top", fontsize=7.6,
                  color=TICK_COLOR, fontweight="bold")

    sax.scatter([log2_scale], [0.5], s=140, color=HANDLE_COLOR, edgecolor="white",
                linewidth=1.4, zorder=3, clip_on=False)


def render(log2_scale):
    frac_zero, mags = compute_frac_zero(log2_scale)
    log_mags = np.log10(np.clip(mags, 1e-300, None))

    fig, ax = plt.subplots(figsize=(7.6, 5.7), dpi=120)
    fig.subplots_adjust(top=0.86, bottom=0.20, left=0.12, right=0.97)

    ax.hist(log_mags, bins=24, range=(Y_MIN, Y_MAX), color=BAR_COLOR, edgecolor="white", linewidth=0.4)
    ax.set_xlim(Y_MIN, Y_MAX)
    ax.set_ylim(0, 25)
    xticks = np.arange(Y_MIN, Y_MAX + 0.01, 2.0)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"$10^{{{t:.0f}}}$" for t in xticks], fontsize=8)
    ax.set_xlabel("first-layer gradient magnitude, after scaling", fontsize=9.5, color=TEXT_COLOR)
    ax.set_ylabel("count (of 64 weights)", fontsize=9.5, color=TEXT_COLOR)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    floor_normal = np.log10(FP16_NORMAL_FLOOR)
    floor_subnormal = np.log10(FP16_SUBNORMAL_FLOOR)
    ax.axvline(floor_normal, color=FLOOR_COLOR, linewidth=1.4, linestyle="--", zorder=3)
    ax.text(floor_normal, ax.get_ylim()[1] * 0.95, " fp16 normal floor", ha="left", va="top",
            fontsize=7.6, color=FLOOR_COLOR, rotation=90)
    ax.axvline(floor_subnormal, color=FLOOR_COLOR, linewidth=1.0, linestyle=":", zorder=3)
    ax.text(floor_subnormal, ax.get_ylim()[1] * 0.95, " fp16 subnormal floor", ha="left", va="top",
            fontsize=7.6, color=FLOOR_COLOR, rotation=90)

    scale = 2 ** log2_scale
    scale_str = f"$2^{{{log2_scale:.0f}}}$" if abs(log2_scale - round(log2_scale)) < 1e-6 else f"{scale:.0f}"
    zero_color = ZERO_COLOR if frac_zero > 0 else "#2e9e44"
    ax.set_title(f"loss scale = {scale_str}   —   {frac_zero:.1%} of gradients still underflow to 0.0 in fp16",
                 fontsize=11.5, fontweight="bold", color=TEXT_COLOR if frac_zero == 0 else zero_color, pad=12)

    draw_slider(fig, log2_scale)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


SWEEP = list(np.arange(0.0, 28.01, 1.0))

frames = []
for log2_scale in SWEEP:
    hold = 8
    if any(abs(log2_scale - cp) < 1e-6 for cp in CHECKPOINTS):
        hold = 50
    frames += [render(log2_scale)] * hold
frames += [render(SWEEP[-1])] * 70

frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=55, loop=0)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
