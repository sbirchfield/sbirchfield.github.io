"""Generate batchnorm_demo.gif: an animated companion to the lesson 37
notebook's "Batch normalization" section. Shows the actual computation as a
spreadsheet-like grid of samples (rows) x channels (columns), in three
steps: (1) compute each channel's mean and std by averaging straight down
its column, (2) normalize -- (x - mean) / std, (3) scale and shift by a
learned per-channel gamma/beta. Highlighting the columns (not rows) makes
concrete that batchnorm statistics are per-channel, computed across the
batch dimension. Each step uses a single stat row below the grid (never
more than one at a time), so the connecting arrow only ever has to cross
one empty gap -- it never has to pass through another row's box or text.

Uses a small 8-sample x 4-channel grid (one scalar per sample/channel, a
simplification of the real per-pixel feature maps, but the per-channel
computation is identical). gamma/beta are set to deliberately non-trivial
values (not nn.BatchNorm2d's all-ones/all-zeros init) specifically so the
final scaled-and-shifted step visibly differs from the normalized step.

Run from anywhere:
    python make_batchnorm_demo.py
Output:
    batchnorm_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import coolwarm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "batchnorm_demo.gif")

TEXT_COLOR = "#222222"
HIGHLIGHT = "#d62728"
GRID_LINE = "#888888"
STAT_FACE = "#eeeeee"

torch.manual_seed(2)
N, C = 8, 4
x = torch.randn(N, C)
x[:, 0] = x[:, 0] * 0.7 - 1.2
x[:, 1] = x[:, 1] * 1.4 + 0.6
x[:, 2] = x[:, 2] * 0.5 + 1.6
x[:, 3] = x[:, 3] * 2.0 - 0.3
x = x.numpy()

# deliberately non-trivial learned gamma/beta (not BatchNorm2d's default
# init of ones/zeros), so the final affine step is visibly different from
# the normalized step -- this is the point being illustrated.
gamma = np.array([1.6, 0.5, 2.2, 0.8])
beta = np.array([1.0, -1.5, 0.3, -0.8])
eps = 1e-5

mean = x.mean(axis=0)
std = x.std(axis=0)
normalized = (x - mean) / (std + eps)
scaled = normalized * gamma + beta

CHANNEL_LABELS = [f"ch {c}" for c in range(C)]
ROW_LABELS = [f"sample {n}" for n in range(N)]

# ---- layout geometry ---------------------------------------------------
CELL_W, CELL_H = 1.55, 0.46
GRID_LEFT = 1.7
GRID_TOP = N * CELL_H + 1.0          # y of the grid's top edge
GRID_BOTTOM = GRID_TOP - N * CELL_H
STAT_GAP = 0.45                      # gap left clear for the connecting arrow
STAT_H = 0.70                        # tall enough for two stacked value lines


def cell_xy(i, j):
    """Top-left corner of grid cell (row i, col j)."""
    return GRID_LEFT + j * CELL_W, GRID_TOP - (i + 1) * CELL_H


def draw_grid(ax, data, col_highlight=False):
    vmax = max(abs(data.min()), abs(data.max()), 1e-6) * 1.05
    norm = Normalize(vmin=-vmax, vmax=vmax)
    for i in range(N):
        for j in range(C):
            x0, y0 = cell_xy(i, j)
            val = data[i, j]
            ax.add_patch(Rectangle((x0, y0), CELL_W, CELL_H,
                                     facecolor=coolwarm(norm(val)),
                                     edgecolor=GRID_LINE, linewidth=0.6, zorder=2))
            ax.text(x0 + CELL_W / 2, y0 + CELL_H / 2, f"{val:+.2f}",
                     ha="center", va="center", fontsize=7.3, color="black", zorder=3)
    if col_highlight:
        for j in range(C):
            x0, _ = cell_xy(0, j)
            ax.add_patch(Rectangle((x0, GRID_BOTTOM), CELL_W, N * CELL_H, fill=False,
                                     edgecolor=HIGHLIGHT, linewidth=2.4, zorder=5))
    # column labels (top) and row labels (left)
    for j, lab in enumerate(CHANNEL_LABELS):
        x0, _ = cell_xy(0, j)
        ax.text(x0 + CELL_W / 2, GRID_TOP + 0.14, lab, ha="center", va="bottom",
                 fontsize=8.5, color=TEXT_COLOR, fontweight="bold")
    for i, lab in enumerate(ROW_LABELS):
        x0, y0 = cell_xy(i, 0)
        ax.text(x0 - 0.12, y0 + CELL_H / 2, lab, ha="right", va="center",
                 fontsize=7.3, color=TEXT_COLOR)


STAT_Y0 = GRID_BOTTOM - STAT_GAP - STAT_H


def draw_stat_block(ax, label_a, values_a, label_b, values_b, arrow_dir):
    """A single stat row below the grid, each cell holding two stacked
    values (e.g. mean & std, or gamma & beta). Only one row ever exists at
    a time, so its connecting arrow only crosses the one clear gap between
    the grid and this row -- it never has to pass through another box."""
    for j in range(C):
        x0 = GRID_LEFT + j * CELL_W
        ax.add_patch(Rectangle((x0, STAT_Y0), CELL_W, STAT_H,
                                 facecolor=STAT_FACE, edgecolor=HIGHLIGHT,
                                 linewidth=2.2, zorder=4))
        ax.text(x0 + CELL_W / 2, STAT_Y0 + STAT_H * 0.68,
                 f"{label_a} = {values_a[j]:+.2f}", ha="center", va="center",
                 fontsize=7.6, color=TEXT_COLOR, zorder=5)
        ax.text(x0 + CELL_W / 2, STAT_Y0 + STAT_H * 0.30,
                 f"{label_b} = {values_b[j]:+.2f}", ha="center", va="center",
                 fontsize=7.6, color=TEXT_COLOR, zorder=5)
        xc = x0 + CELL_W / 2
        if arrow_dir == "down":
            y_from, y_to = GRID_BOTTOM - 0.04, STAT_Y0 + STAT_H + 0.04
        else:
            y_from, y_to = STAT_Y0 + STAT_H + 0.04, GRID_BOTTOM - 0.04
        ax.annotate("", xy=(xc, y_to), xytext=(xc, y_from),
                     arrowprops=dict(arrowstyle="-|>", color=HIGHLIGHT, lw=1.6), zorder=6)


STAGES = [
    dict(data=x, title="raw activations x", formula=None,
         subtitle=["each row is a sample, each column a channel",
                    "8 samples in the batch, 4 channels"],
         stat=None, col_highlight=False),
    dict(data=x, title="step 1: compute each channel's mean and std", formula=None,
         subtitle="averaged straight down each column, across the batch",
         stat=dict(label_a="μ", values_a=mean, label_b="σ", values_b=std, arrow_dir="down"),
         col_highlight=True),
    dict(data=normalized, title="step 2: normalize", formula=r"$(x-\mu)/\sigma$",
         subtitle="every channel now has mean 0, std 1",
         stat=dict(label_a="μ", values_a=mean, label_b="σ", values_b=std, arrow_dir="up"),
         col_highlight=True),
    dict(data=scaled, title="step 3: scale and shift", formula=r"$\gamma\cdot\hat{x}+\beta$",
         subtitle="a learned per-channel affine can undo the normalization",
         stat=dict(label_a="γ", values_a=gamma, label_b="β", values_b=beta, arrow_dir="up"),
         col_highlight=True),
]

# Y_MAX only needs enough headroom for the column-label text drawn just
# above the grid (GRID_TOP + 0.14) -- no extra buffer beyond that, since
# the title/formula/subtitle live outside the axes entirely (see below).
Y_MIN = STAT_Y0 - 0.15
Y_MAX = GRID_TOP + 0.26
FIG_H = Y_MAX - Y_MIN   # axes occupies exactly this many inches -- 1 data unit = 1 inch

# Text zone above the axes, sized in inches and kept separate from FIG_H so
# the grid/cell geometry above is never silently rescaled by how much text
# a given stage needs.
BOTTOM_MARGIN = 0.10
TITLE_OFFSET = 0.22   # inches below the figure's top edge
LINE2_OFFSET = 0.52
LINE3_OFFSET = 0.76
TEXT_H = LINE3_OFFSET + 0.10


def render(stage_idx):
    stage = STAGES[stage_idx]
    total_h = FIG_H + TEXT_H + BOTTOM_MARGIN
    fig, ax = plt.subplots(figsize=(6.6, total_h), dpi=120)

    draw_grid(ax, stage["data"], col_highlight=stage["col_highlight"])
    if stage["stat"] is not None:
        draw_stat_block(ax, **stage["stat"])

    ax.set_xlim(GRID_LEFT - 1.9, GRID_LEFT + C * CELL_W + 0.3)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.axis("off")

    y_title = 1 - TITLE_OFFSET / total_h
    y_line2 = 1 - LINE2_OFFSET / total_h
    y_line3 = 1 - LINE3_OFFSET / total_h

    fig.suptitle(stage["title"], fontsize=12, color=TEXT_COLOR, fontweight="bold", y=y_title)

    # up to two more lines (formula, then subtitle line(s)), each in its own
    # fixed slot -- never more than one line per slot, so no line's height
    # can eat into the next slot's reserved space.
    subtitle_lines = stage["subtitle"]
    if isinstance(subtitle_lines, str):
        subtitle_lines = [subtitle_lines]
    elif subtitle_lines is None:
        subtitle_lines = []
    extra_lines = []
    if stage["formula"] is not None:
        extra_lines.append((stage["formula"], 12, HIGHLIGHT, "normal"))
    extra_lines += [(s, 9.5, "#555555", "italic") for s in subtitle_lines]
    for i, (text, fontsize, color, style) in enumerate(extra_lines[:2]):
        y = y_line2 if i == 0 else y_line3
        fig.text(0.5, y, text, ha="center", fontsize=fontsize, color=color, style=style)

    top_frac = 1 - TEXT_H / total_h
    bottom_frac = BOTTOM_MARGIN / total_h
    fig.subplots_adjust(top=top_frac, bottom=bottom_frac, left=0.02, right=0.98)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD = 85
for step in range(len(STAGES)):
    frames += [render(step)] * HOLD

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=55,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
