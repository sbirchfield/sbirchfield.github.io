"""Generate haar_cascade_demo.gif: an animated companion to the lesson 16
notebook's "The Haar wavelet" cell. A 1D signal is recursively split into an
approximation (local averages) and a detail (local differences) using the
notebook's own Haar formulas,

    a_k = (x_2k + x_2k+1) / sqrt(2),   d_k = (x_2k - x_2k+1) / sqrt(2)

across 3 levels. Every array gets its own fixed, dedicated panel in a grid
(original signal on top, one approximation|detail column pair per level
below) -- panels are revealed and highlighted as the cascade proceeds, but
never reused for different data, so the original signal stays visible the
whole time and no panel's horizontal scale ever changes mid-animation.

Within each level, a highlighted window of 2 input samples slides across the
source panel (non-overlapping pairs), and the corresponding approximation
and detail bars grow in one pair at a time, directly underneath -- making
the "2-tap filter slides across the signal" mechanics of the Haar transform
visible instead of just showing the finished a/d arrays appear all at once.

Run from anywhere:
    python make_haar_cascade_demo.py
Output:
    haar_cascade_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "haar_cascade_demo.gif")

MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"
DETAIL_COLOR = "#6a6a6a"
HIGHLIGHT_COLOR = "#2f6fb0"  # a single, consistent accent for "this is what's
# being combined right now" -- used for both the sliding source window and
# the newly-added output bar's outline, instead of a separate alert color.

rng = np.random.default_rng(3)
N0 = 16
x = np.linspace(0, 3 * np.pi, N0)
signal = 5 * np.sin(x) + 0.6 * rng.standard_normal(N0) + np.where(np.arange(N0) > 10, 3.0, 0.0)

NUM_LEVELS = 3
approxs, details = [], []
cur = signal
for _ in range(NUM_LEVELS):
    a = (cur[0::2] + cur[1::2]) / np.sqrt(2)
    d = (cur[0::2] - cur[1::2]) / np.sqrt(2)
    approxs.append(a)
    details.append(d)
    cur = a

YLIM = (signal.min() - 2, signal.max() + 2)


def axes_style(ax, n):
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(*YLIM)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)


def stem_on(ax, vals, color, label, filled=None, emphasize_idx=None):
    n = len(vals)
    axes_style(ax, n)
    filled = n if filled is None else filled
    for i in range(filled):
        is_new = i == emphasize_idx
        ax.bar(i, vals[i], width=0.7, color=color,
                edgecolor=(HIGHLIGHT_COLOR if is_new else "none"),
                linewidth=(2.2 if is_new else 0), zorder=2)
    # title sits above the axes, outside the plot area, so it can never
    # collide with a tall bar the way an in-plot text label could
    ax.set_title(label, fontsize=9.5, color=color, pad=6)


def placeholder(ax, n_full, label):
    """Faint dashed outline for a not-yet-computed panel, so the full grid
    structure is visible from the first frame instead of looking like blank
    empty space."""
    axes_style(ax, n_full)
    ax.add_patch(plt.Rectangle((0.02, 0.06), 0.96, 0.88, transform=ax.transAxes,
                                 fill=False, edgecolor="#dddddd", linewidth=1.0, linestyle="--"))
    ax.set_title(label, fontsize=9.5, color="#cccccc", pad=6)


def highlight_window(ax, n, k):
    """Translucent box around the pair of input samples (indices 2k, 2k+1)
    currently being combined into one a/d output -- the 2-tap Haar filter's
    sliding window, drawn behind the bars it spans."""
    ax.add_patch(plt.Rectangle((2 * k - 0.5, YLIM[0]), 2, YLIM[1] - YLIM[0],
                                 facecolor=HIGHLIGHT_COLOR, alpha=0.15,
                                 edgecolor=HIGHLIGHT_COLOR, linewidth=1.3, zorder=1))


def render(level, filled, emphasize_idx):
    """level: which level (0..NUM_LEVELS-1) is currently being split.
    filled: how many of that level's a/d pairs are already computed.
    emphasize_idx: index of the pair just added this frame (None once settled).
    Levels before `level` are fully done; levels after are placeholders."""
    fig = plt.figure(figsize=(8.5, 6.6), dpi=120)
    gs = fig.add_gridspec(NUM_LEVELS + 1, 2, hspace=0.55, wspace=0.15,
                            top=0.89, bottom=0.06, left=0.08, right=0.97)

    ax_orig = fig.add_subplot(gs[0, :])
    stem_on(ax_orig, signal, NEUTRAL_COLOR, "original signal")
    if level == 0 and emphasize_idx is not None:
        highlight_window(ax_orig, N0, emphasize_idx)

    approx_axes = []
    for L in range(NUM_LEVELS):
        row = L + 1
        ax_a = fig.add_subplot(gs[row, 0])
        ax_d = fig.add_subplot(gs[row, 1])
        n_out = len(approxs[L])

        if L < level:
            stem_on(ax_a, approxs[L], MATCH_COLOR, f"level {L + 1} approx")
            stem_on(ax_d, details[L], DETAIL_COLOR, f"level {L + 1} detail")
        elif L == level:
            stem_on(ax_a, approxs[L], MATCH_COLOR, f"level {L + 1} approx", filled=filled, emphasize_idx=emphasize_idx)
            stem_on(ax_d, details[L], DETAIL_COLOR, f"level {L + 1} detail", filled=filled, emphasize_idx=emphasize_idx)
            if emphasize_idx is not None and L > 0:
                highlight_window(approx_axes[L - 1], len(approxs[L - 1]), emphasize_idx)
        else:
            placeholder(ax_a, n_out, f"level {L + 1} approx")
            placeholder(ax_d, n_out, f"level {L + 1} detail")

        approx_axes.append(ax_a)

    fig.suptitle("Haar cascade: slide filters across the signal, producing approx + detail",
                   fontsize=12, color=NEUTRAL_COLOR, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_INITIAL = 16
HOLD_STEP = 11
HOLD_SETTLE = 14
HOLD_END = 44

frames += [render(level=0, filled=0, emphasize_idx=None)] * HOLD_INITIAL

for L in range(NUM_LEVELS):
    n_out = len(approxs[L])
    for k in range(n_out):
        frames += [render(level=L, filled=k + 1, emphasize_idx=k)] * HOLD_STEP
    is_last = L == NUM_LEVELS - 1
    frames += [render(level=L, filled=n_out, emphasize_idx=None)] * (HOLD_END if is_last else HOLD_SETTLE)

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=90,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
