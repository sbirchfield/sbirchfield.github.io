"""Generate pooling_demo.gif: an animated companion to the lesson 34
notebook's "Pooling: downsampling with a purpose" section. Steps a 2x2,
stride-2 max-pooling window across a small feature map, highlighting the
window and marking which cell holds the max at each stop, while the
downsampled output grid fills in -- making concrete what maxpool2d() does:
keep only the largest value in each non-overlapping window.

Run from anywhere:
    python make_pooling_demo.py
Output:
    pooling_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "pooling_demo.gif")

ACTIVE_COLOR = "#e07b00"
DONE_COLOR = "#3a6fb0"
MAX_COLOR = "#2e9e44"

feature_map = np.array([
    [1, 7, 2, 1],
    [2, 3, 6, 0],
    [0, 2, 9, 1],
    [1, 8, 0, 3],
], dtype=np.float64)

SIZE = 2
H, W = feature_map.shape
OH, OW = H // SIZE, W // SIZE

windows = [(i, j) for i in range(OH) for j in range(OW)]


def window_argmax(i, j):
    patch = feature_map[i * SIZE:i * SIZE + SIZE, j * SIZE:j * SIZE + SIZE]
    local = np.unravel_index(np.argmax(patch), patch.shape)
    return i * SIZE + local[0], j * SIZE + local[1], patch.max()


full_output = np.array([[window_argmax(i, j)[2] for j in range(OW)] for i in range(OH)])


def draw_input_grid(ax, step):
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="#aaaaaa", linewidth=0.7)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.tick_params(length=0, labelsize=7)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)

    is_stepping = step < len(windows)
    i, j = windows[min(step, len(windows) - 1)]
    mr, mc, mval = window_argmax(i, j)
    all_max_rc = {window_argmax(wi, wj)[:2] for wi, wj in windows}

    for r in range(H):
        for c in range(W):
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="white",
                                     edgecolor="none", zorder=0))
            is_max_here = (is_stepping and (r, c) == (mr, mc)) or \
                (not is_stepping and (r, c) in all_max_rc)
            color = MAX_COLOR if is_max_here else "#222222"
            weight = "bold" if is_max_here else "normal"
            ax.text(c, r, f"{feature_map[r, c]:g}", ha="center", va="center",
                     fontsize=10, color=color, fontweight=weight, zorder=1)

    if is_stepping:
        ax.add_patch(Rectangle((j * SIZE - 0.5, i * SIZE - 0.5), SIZE, SIZE,
                                 facecolor=ACTIVE_COLOR, alpha=0.22,
                                 edgecolor=ACTIVE_COLOR, linewidth=3.0, zorder=3))
        ax.add_patch(Rectangle((mc - 0.5, mr - 0.5), 1, 1, fill=False,
                                 edgecolor=MAX_COLOR, linewidth=2.4, zorder=4))
    else:
        for r, c in all_max_rc:
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                     facecolor=MAX_COLOR, alpha=0.20,
                                     edgecolor=MAX_COLOR, linewidth=2.0, zorder=0.5))

    ax.set_title("feature map (4x4)", fontsize=12)


def draw_output_grid(ax, computed_mask, current_rc=None):
    for r in range(OH):
        for c in range(OW):
            if current_rc == (r, c):
                face, text_color, val = "#eafaea", MAX_COLOR, full_output[r, c]
                show = True
            elif computed_mask[r, c]:
                face, text_color, val = "#eef3fa", DONE_COLOR, full_output[r, c]
                show = True
            else:
                face, text_color, val = "#ffffff", None, None
                show = False
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=face,
                                     edgecolor="#aaaaaa", linewidth=0.7, zorder=0))
            if show:
                ax.text(c, r, f"{val:g}", ha="center", va="center", fontsize=11,
                         color=text_color, fontweight="bold", zorder=1)
            if current_rc == (r, c):
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                         edgecolor=MAX_COLOR, linewidth=2.6, zorder=2))

    ax.set_xticks(range(OW))
    ax.set_yticks(range(OH))
    ax.tick_params(length=0, labelsize=7)
    ax.set_xlim(-0.5, OW - 0.5)
    ax.set_ylim(OH - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    ax.set_title("pooled output (2x2)", fontsize=12)


def render(step):
    total = len(windows)
    is_stepping = step < total

    fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(8.6, 4.6), dpi=120,
                                          gridspec_kw={"width_ratios": [1.0, 0.65]})
    fig.subplots_adjust(top=0.8, bottom=0.08, wspace=0.35)

    draw_input_grid(ax_in, step)

    computed_mask = np.zeros((OH, OW), dtype=bool)
    flat = min(step, total)
    for k in range(flat):
        i, j = windows[k]
        computed_mask[i, j] = True
    current_rc = windows[step] if is_stepping else None
    draw_output_grid(ax_out, computed_mask, current_rc=current_rc)

    if is_stepping:
        i, j, val = window_argmax(*windows[step])
        title = f"window ({i // SIZE}, {j // SIZE}): keep the max, {val:g}"
        color = ACTIVE_COLOR
    else:
        title = "every window shrinks to its single largest value"
        color = DONE_COLOR
    fig.suptitle(title, fontsize=12.5, color=color, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 55
HOLD_END = 90
total = len(windows)
for step in range(total + 1):
    frames += [render(step)] * (HOLD_END if step == total else HOLD_STEP)

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=55,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames); output=\n{full_output}")
