"""Generate convolution_demo.gif: an animated, step-by-step walkthrough of 1D
convolution by hand -- flip the kernel, slide it across the signal, and at
each position take the dot product with the overlapping window.

Uses the same signal and (pre-flipped) kernel as the lesson 10 notebook's
`convolve1d` example, so the printed output values match exactly.

Run from anywhere:
    python make_convolution_demo.py
Output:
    convolution_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "convolution_demo.gif")

# Same signal and kernel as the lesson 10 notebook's convolve1d() example.
signal = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=np.float64)
kernel = 0.5 * np.array([1, 0, -1], dtype=np.float64)  # as passed to convolve1d
flipped = kernel[::-1]  # what actually gets dotted against each window

N = len(signal)
KSIZE = len(kernel)
PAD = KSIZE // 2
padded = np.pad(signal, PAD, mode="constant")

ACTIVE_COLOR = "#e07b00"   # orange: the position/output value currently being computed
DONE_COLOR = "#3a6fb0"     # blue: output values already computed
SIGNAL_COLOR = "#444444"
PAD_COLOR = "#cccccc"

Y_MIN, Y_MAX = -3.5, 6.6


def output_at(n):
    window = padded[n:n + KSIZE]
    return float(np.dot(window, flipped))


full_output = np.array([output_at(n) for n in range(N)])


def draw_value_bars(ax, xs, values, color, base, box_h, box_w, fontsize,
                     centerline_pad):
    """Draw a horizontal gray centerline through `base`, then for each (x,
    value) an orange box starting at the centerline and stretching up (v >
    0) or down (v < 0), with the value labeled just outside the box --
    matching the look of the signal bar chart."""
    ax.hlines(base, xs.min() - centerline_pad, xs.max() + centerline_pad,
               color="#dddddd", linewidth=0.8, zorder=0)
    label_gap = box_h * 0.18
    for x, v in zip(xs, values):
        if v > 0:
            y0, y1 = base, base + box_h
        elif v < 0:
            y0, y1 = base - box_h, base
        else:
            y0, y1 = base - box_h * 0.04, base + box_h * 0.04
        ax.add_patch(Rectangle((x - box_w / 2, y0), box_w, y1 - y0, fill=True,
                                facecolor="white", edgecolor=color, linewidth=1.6, zorder=2))
        if v > 0:
            label_y, va = y1 + label_gap, "bottom"
        elif v < 0:
            label_y, va = y0 - label_gap, "top"
        else:
            label_y, va = base, "center"
        ax.text(x, label_y, f"{v:g}", ha="center", va=va, fontsize=fontsize,
                 color=color, fontweight="bold", zorder=3)


def draw_kernel_panel(ax):
    """Static panel shown every frame: the raw kernel, an arrow showing it
    gets flipped end-to-end, and the flipped kernel that is actually used
    in the sliding dot product. Each kernel is drawn the same way as the
    signal: a gray centerline with orange boxes above/below it."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    xs = np.linspace(0.3, 0.86, len(kernel))
    draw_value_bars(ax, xs, kernel, "#555555", base=0.80,
                     box_h=0.085, box_w=0.14, fontsize=9, centerline_pad=0.08)
    ax.annotate("", xy=(0.58, 0.56), xytext=(0.58, 0.70),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.6))
    ax.text(0.42, 0.63, "flip", ha="right", va="center", fontsize=11,
             color="#555555", style="italic")
    draw_value_bars(ax, xs, flipped, ACTIVE_COLOR, base=0.36,
                     box_h=0.085, box_w=0.14, fontsize=9, centerline_pad=0.08)
    ax.set_title("kernel", fontsize=12)


def draw_signal_bars(ax, highlight_n=None):
    xs = np.arange(-PAD, N + PAD)
    heights = padded
    for x, h in zip(xs, heights):
        is_pad = x < 0 or x >= N
        color = PAD_COLOR if is_pad else SIGNAL_COLOR
        ax.bar(x, h, width=0.6, color=color, zorder=2,
                edgecolor="none" if not is_pad else "#999999",
                linestyle="--" if is_pad else "-")
        if not is_pad:
            ax.text(x, h + 0.25, f"{h:g}", ha="center", va="bottom", fontsize=8, color="#333333")

    if highlight_n is not None:
        win_xs = np.arange(highlight_n - PAD, highlight_n + PAD + 1)
        left, right = win_xs[0] - 0.45, win_xs[-1] + 0.45
        ax.add_patch(Rectangle((left, Y_MIN + 0.1), right - left, Y_MAX - Y_MIN - 0.2,
                                fill=False, edgecolor=ACTIVE_COLOR, linewidth=2.4, zorder=1))

        draw_value_bars(ax, win_xs, flipped, ACTIVE_COLOR, base=Y_MIN + 1.15,
                         box_h=0.55, box_w=0.5, fontsize=9, centerline_pad=0.45)

    ax.axhline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_xlim(-PAD - 0.7, N + PAD - 0.3)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks(range(0, N))
    ax.set_yticks([])
    ax.set_title("signal (gray = zero padding)", fontsize=12)


def draw_output_bars(ax, computed_upto, current_n=None):
    xs = np.arange(N)
    for x in xs:
        if current_n is not None and x == current_n:
            h, color = full_output[x], ACTIVE_COLOR
        elif x < computed_upto:
            h, color = full_output[x], DONE_COLOR
        else:
            h, color = 0.0, None
        if color is not None:
            ax.bar(x, h, width=0.6, color=color, zorder=2)
            ax.text(x, h + (0.25 if h >= 0 else -0.35), f"{h:g}",
                     ha="center", va="bottom" if h >= 0 else "top", fontsize=8, color="#333333")

    ax.axhline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_xlim(-PAD - 0.7, N + PAD - 0.3)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks(range(0, N))
    ax.set_yticks([])
    ax.set_title("output = signal * kernel", fontsize=12)


def render(step, final_hold=False):
    """step: -1 for the blank intro (no window shown yet), 0..N-1 for the
    position currently being computed, or N once every position is done."""
    is_stepping = 0 <= step < N

    fig, (ax_kernel, ax_signal, ax_output) = plt.subplots(
        1, 3, figsize=(14.0, 4.8), dpi=120,
        gridspec_kw={"width_ratios": [0.42, 1.0, 1.0]},
    )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.78, bottom=0.1, wspace=0.22)

    draw_kernel_panel(ax_kernel)
    draw_signal_bars(ax_signal, highlight_n=step if is_stepping else None)
    draw_output_bars(ax_output, computed_upto=max(step, 0), current_n=step if is_stepping else None)

    if is_stepping:
        terms = " + ".join(f"({k:g})({padded[step + i]:g})" for i, k in enumerate(flipped))
        suptitle = f"position n={step}: output[{step}] = {terms} = {full_output[step]:g}"
        color = ACTIVE_COLOR
    elif step < 0:
        suptitle = "flipped kernel will slide across the zero-padded signal"
        color = "#222222"
    else:
        suptitle = ""
        color = "#222222"

    if suptitle:
        fig.suptitle(suptitle, fontsize=12.5, color=color)

    if not is_stepping and step >= 0:
        ax_output.text(0.5, 1.2, "FINAL RESULT", transform=ax_output.transAxes,
                         ha="center", va="bottom", fontsize=15, fontweight="bold",
                         color="#1f1f1f")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_INTRO = 10
HOLD_STEP = 14
HOLD_END = 28

frames += [render(-1)] * HOLD_INTRO

for step in range(N):
    frames += [render(step)] * HOLD_STEP

frames += [render(N, final_hold=True)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=110,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames, {N} positions)")
