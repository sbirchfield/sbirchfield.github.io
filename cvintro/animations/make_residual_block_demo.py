"""Generate residual_block_demo.gif: an animated companion to the lesson 37
notebook's "Putting it together: a convolutional residual block" section.
A schematic data-flow diagram contrasting a plain conv stack with a
residual block: a highlighted token steps through conv1 -> bn1 -> relu ->
conv2 -> bn2 -> (+) -> relu, identical in both rows, except the residual
row also has a skip-connection arrow carrying the block's input straight
to the final addition -- the only structural difference between the two.

Matches the notebook's own ResidualBlock exactly:
    out = relu(bn1(conv1(x)))
    out = bn2(conv2(out))
    return relu(x + out)   # residual row only; plain row has no "+ x"

Run from anywhere:
    python make_residual_block_demo.py
Output:
    residual_block_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "residual_block_demo.gif")

BOX_COLOR = "#e8eef7"
ACTIVE_COLOR = "#f2c200"
SKIP_COLOR = "#2e9e44"
TEXT_COLOR = "#222222"
EDGE_COLOR = "#555555"

STAGES = ["x", "conv1", "bn1", "relu", "conv2", "bn2", "+", "relu"]
STAGE_DESC = ["input", "convolution", "batch norm", "ReLU", "convolution", "batch norm",
              "addition", "ReLU"]
N = len(STAGES)
XS = [i * 1.3 for i in range(N)]


def draw_row(ax, y, active_idx, with_skip):
    # the plain row has no addition step at all -- skip drawing a box for
    # "+" entirely, and connect bn2 straight to the final relu, so the
    # missing operation is visually obvious rather than shown as an empty box.
    indices = range(N) if with_skip else [i for i, s in enumerate(STAGES) if s != "+"]
    prev_i = None
    for i in indices:
        x, label = XS[i], STAGES[i]
        is_active = (i == active_idx)
        face = ACTIVE_COLOR if is_active else BOX_COLOR
        w, h = 0.95, 0.55
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=face,
                                 edgecolor=EDGE_COLOR, linewidth=1.3, zorder=2))
        ax.text(x, y, label, ha="center", va="center", fontsize=9.5,
                 color=TEXT_COLOR, fontweight="bold" if is_active else "normal", zorder=3)
        if prev_i is not None:
            prev_x, prev_w = XS[prev_i], 0.95
            ax.annotate("", xy=(x - w / 2, y), xytext=(prev_x + prev_w / 2, y),
                         arrowprops=dict(arrowstyle="->", color=EDGE_COLOR, lw=1.3), zorder=1)
        prev_i = i

    if with_skip:
        add_x = XS[STAGES.index("+")]
        skip_active = (active_idx == STAGES.index("+"))
        lw = 2.4 if skip_active else 1.3
        box_top = y + 0.28
        skip_y = y + 0.85
        # a single continuous U-shaped connector (one FancyArrowPatch via
        # the "bar" connection style) instead of three separate segments --
        # no visible seams where the pieces used to meet.
        frac = -(skip_y - box_top) / (add_x - XS[0])
        ax.annotate("", xy=(add_x, box_top), xytext=(XS[0], box_top),
                     arrowprops=dict(arrowstyle="->", color=EDGE_COLOR, lw=lw,
                                       connectionstyle=f"bar,fraction={frac},armA=0,armB=0"),
                     zorder=4)
        label_color = SKIP_COLOR if skip_active else TEXT_COLOR
        ax.text((XS[0] + add_x) / 2, skip_y + 0.2, "skip connection: + x", ha="center",
                 fontsize=8.5, color=label_color, fontweight="bold")


# steps: highlight each stage in turn (0..N-1); skip connection is always
# drawn in the residual row, and glows brighter exactly when the "+" stage
# is reached, since that's the moment it actually contributes.
def render(step):
    active_idx = min(step, N - 1)

    fig, ax = plt.subplots(figsize=(9.6, 5.0), dpi=120)
    fig.subplots_adjust(top=0.80, bottom=0.08, left=0.14, right=0.97)

    y_plain, y_res = 1.0, 3.2
    draw_row(ax, y_res, active_idx, with_skip=True)
    draw_row(ax, y_plain, active_idx, with_skip=False)

    ax.text(-0.9, y_res, "residual", ha="right", va="center", fontsize=10.5,
             color=TEXT_COLOR, fontweight="bold")
    ax.text(-0.9, y_plain, "plain", ha="right", va="center", fontsize=10.5,
             color=TEXT_COLOR, fontweight="bold")

    ax.set_xlim(-2.0, XS[-1] + 1.0)
    ax.set_ylim(0.1, 4.6)
    ax.axis("off")

    stage_name = STAGES[active_idx]
    desc = STAGE_DESC[active_idx]
    if stage_name == "+":
        title = "the shortcut rejoins here: output = relu(x + F(x))"
        subtitle = ""
    else:
        title = f"stage: {desc}"
        subtitle = "" if stage_name == "x" else "(same computation in both rows)"
    fig.suptitle(title, fontsize=12, color=TEXT_COLOR, fontweight="bold", y=0.95)
    if subtitle:
        fig.text(0.5, 0.875, subtitle, ha="center", fontsize=9.5, color="#555555", style="italic")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 55
HOLD_END = 90
for step in range(N):
    frames += [render(step)] * (HOLD_END if step == N - 1 else HOLD_STEP)

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=55,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
