"""Generate finetune_lr_demo.gif: an animated companion to the lesson 38
notebook's "Three strategies" section. Shows all three strategies stacked
top to bottom for direct comparison, in four beats: (1) a forward pass,
data flowing left to right, identical for all three strategies; (2) a brief
reveal of every layer's frozen/trainable status (snowflake vs flame, sized
by learning rate), held for a moment before backprop starts; (3) a backward
pass, gradients flowing right to left, where only layers that actually get
a weight update are marked -- for the frozen-backbone strategy, that means
nothing past the head ever gets marked, since nothing behind it needs to
change; (4) a final hold. Fire is red/orange, snowflake is light blue,
matching the usual convention. A legend stays visible throughout rather
than only appearing at the end.

Reuses the notebook's own learning rates exactly: lr=0.01 for the head in
all three strategies, backbone_lr=0.0003 for fine-tuning (roughly 30x
smaller than the head's, as the book text notes) -- flame size is scaled
directly from these numbers, not just illustrative.

Run from anywhere:
    python make_finetune_lr_demo.py
Output:
    finetune_lr_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "finetune_lr_demo.gif")

BOX_COLOR = "#e8eef7"
HEAD_COLOR = "#fff4d6"
UPDATED_FACE = "#ffd9d0"
TEXT_COLOR = "#222222"
EDGE_COLOR = "#999999"
FORWARD_COLOR = "#1f6fb4"
FORWARD_FAINT = "#b9cfe6"
BACKWARD_COLOR = "#d62728"
BACKWARD_FAINT = "#eeb9ae"
SNOW_COLOR = "#6fb8e8"
FIRE_COLOR = "#e8540c"

HEAD_LR = 0.01
BACKBONE_FINETUNE_LR = 0.0003
HEAD_ICON_SIZE = 20
FINETUNE_ICON_SIZE = max(HEAD_ICON_SIZE * (BACKBONE_FINETUNE_LR / HEAD_LR) ** 0.5, 10.0)
FROZEN_ICON_SIZE = 18

EMOJI_FONT = fm.FontProperties(fname=r"C:\Windows\Fonts\seguiemj.ttf")
SNOWFLAKE = "\u2744"
FIRE = "\U0001F525"

LEGEND_TEXT = ("snowflake = frozen    "
               "flame = trainable (bigger = larger learning rate)    "
               "red box = weights actually updated")

# node 0 = input (no icon, never "updated"), nodes 1-3 = backbone conv
# layers, node 4 = head
NODE_LABELS = ["input", "conv1", "conv2", "conv3", "head"]
NODE_XS = [0.2, 1.7, 3.3, 4.9, 6.7]
N_EDGES = len(NODE_XS) - 1

STRATEGIES = [
    ("from scratch", "backbone trains at the same rate as the head \u2014 nothing pretrained to protect"),
    ("frozen backbone", "backbone locked \u2014 only the new head updates at all"),
    ("fine-tuned", f"backbone updates too, but ~30x slower ({BACKBONE_FINETUNE_LR} vs {HEAD_LR}) \u2014 small corrections, not a rewrite"),
]


def node_style(strategy_name, node_idx):
    """Returns (frozen, icon_size) for backbone nodes 1-3 and the head
    node 4. node 0 (input) is never queried."""
    if node_idx == 4:
        return False, HEAD_ICON_SIZE
    if strategy_name == "from scratch":
        return False, HEAD_ICON_SIZE
    if strategy_name == "frozen backbone":
        return True, FROZEN_ICON_SIZE
    return False, FINETUNE_ICON_SIZE  # fine-tuned


BLOCK_H = 1.75
ROW_TOPS = [6.5, 6.5 - BLOCK_H - 0.55, 6.5 - 2 * (BLOCK_H + 0.55)]

# ---- animation timeline -----------------------------------------------
FORWARD_N = N_EDGES       # steps 0..3: forward reaches nodes 1..4
REVEAL_N = 1               # one held beat: every icon appears at once
# backward starts at the head (closest to the loss, updates first) and
# works back toward the input -- the input itself is never "updated".
BACKWARD_ORDER = [4, 3, 2, 1, 0]
BACKWARD_N = len(BACKWARD_ORDER)
HOLD_N = 1


def draw_icon(ax, x, y, frozen, size):
    glyph, color = (SNOWFLAKE, SNOW_COLOR) if frozen else (FIRE, FIRE_COLOR)
    ax.text(x, y, glyph, ha="center", va="center", fontsize=size,
             fontproperties=EMOJI_FONT, color=color, zorder=5)


LANE_DY = 0.10


def draw_row(ax, row_top, strategy_idx, phase, phase_progress):
    strategy_name, caption = STRATEGIES[strategy_idx]
    box_h = 0.62
    y_icon = row_top - 0.55
    y_box = row_top - 1.05
    y_caption = row_top - 1.55

    ax.text(-1.9, row_top, strategy_name, ha="left", va="center", fontsize=11.5,
             color=TEXT_COLOR, fontweight="bold")

    forward_lit = set()        # edges currently being drawn (bold) during the forward phase
    backward_lit = set()       # edges already crossed by the backward sweep
    current_node = None        # node the active sweep is at right now
    current_dir = None         # "forward" or "backward", for the spotlight outline
    icons_shown = phase != "forward"
    updated_nodes = set()      # nodes whose weights have actually been updated so far

    if phase == "forward":
        forward_lit = set(range(phase_progress + 1))
        current_node = phase_progress + 1
        current_dir = "forward"
    elif phase == "reveal":
        forward_lit = set(range(N_EDGES))
    elif phase == "backward":
        forward_lit = set(range(N_EDGES))
        visited = BACKWARD_ORDER[:phase_progress + 1]
        current_node = visited[-1]
        backward_lit = set(range(current_node, 4))
        current_dir = "backward"
        updated_nodes = {n for n in visited if n != 0 and not node_style(strategy_name, n)[0]}
    else:  # final hold
        forward_lit = set(range(N_EDGES))
        backward_lit = set(range(N_EDGES))
        updated_nodes = {n for n in range(1, 5) if not node_style(strategy_name, n)[0]}

    forward_done = phase in ("reveal", "backward", "final")

    # draw edges as two parallel lanes: forward (upper) and backward (lower)
    # -- so both directions stay visible at once instead of one replacing
    # the other, and the backward lane always renders below the forward one.
    for i in range(N_EDGES):
        x_fwd_tail, x_fwd_head = NODE_XS[i] + 0.1, NODE_XS[i + 1] - 0.55
        if forward_done:
            fcolor, flw = FORWARD_FAINT, 0.9
        else:
            fcolor = FORWARD_COLOR if i in forward_lit else EDGE_COLOR
            flw = 2.2 if i in forward_lit else 1.0
        ax.annotate("", xy=(x_fwd_head, y_box + LANE_DY), xytext=(x_fwd_tail, y_box + LANE_DY),
                     arrowprops=dict(arrowstyle="-|>", color=fcolor, lw=flw,
                                       mutation_scale=14), zorder=2)

        if phase in ("backward", "final"):
            # the backward arrowhead must land exactly at the LEFT node's
            # right box edge (not inside it) or the node's own rectangle
            # (drawn on top, zorder=3) hides the arrowhead entirely.
            x_bwd_tail, x_bwd_head = NODE_XS[i + 1] - 0.55, NODE_XS[i] + 0.55
            bcolor = BACKWARD_COLOR if i in backward_lit else BACKWARD_FAINT
            blw = 2.2 if i in backward_lit else 0.9
            ax.annotate("", xy=(x_bwd_head, y_box - LANE_DY), xytext=(x_bwd_tail, y_box - LANE_DY),
                         arrowprops=dict(arrowstyle="-|>", color=bcolor, lw=blw,
                                           mutation_scale=14), zorder=2)

    # draw nodes
    for i, (x, label) in enumerate(zip(NODE_XS, NODE_LABELS)):
        if i == 0:
            ax.text(x, y_box, "input", ha="right", va="center", fontsize=9.5, color=TEXT_COLOR)
            continue
        frozen, size = node_style(strategy_name, i)
        updated = i in updated_nodes
        face = UPDATED_FACE if updated else (HEAD_COLOR if i == 4 else BOX_COLOR)
        edge_color = BACKWARD_COLOR if updated else EDGE_COLOR
        edge_lw = 1.8 if updated else 1.2
        ax.add_patch(Rectangle((x - 0.55, y_box - box_h / 2), 1.1, box_h, facecolor=face,
                                 edgecolor=edge_color, linewidth=edge_lw, zorder=3))
        # the "sweep is here right now" spotlight: shown for every node
        # during the forward pass (every layer does compute forward,
        # frozen or not), but during backward only for a node that will
        # actually be updated -- a frozen node gets no box treatment at
        # all once backprop reaches it, matching "hashed/highlighted only
        # for layers that actually update."
        show_spotlight = (i == current_node) and (
            current_dir == "forward" or (current_dir == "backward" and not frozen)
        )
        if show_spotlight:
            outline_color = FORWARD_COLOR if current_dir == "forward" else BACKWARD_COLOR
            ax.add_patch(Rectangle((x - 0.55, y_box - box_h / 2), 1.1, box_h, fill=False,
                                     edgecolor=outline_color, linewidth=3.2, zorder=5))
        ax.text(x, y_box, label, ha="center", va="center", fontsize=9, color=TEXT_COLOR,
                 fontweight="bold" if i == 4 else "normal", zorder=4)
        if icons_shown:
            draw_icon(ax, x, y_icon, frozen, size)

    ax.text(2.6, y_caption, caption, ha="center", va="center", fontsize=8, color="#555555",
             style="italic")


def render(step):
    if step < FORWARD_N:
        phase, progress = "forward", step
        banner = "forward pass: data flows left to right"
        banner_color = FORWARD_COLOR
    elif step < FORWARD_N + REVEAL_N:
        phase, progress = "reveal", 0
        banner = "each layer is frozen or trainable \u2014 now backpropagate"
        banner_color = TEXT_COLOR
    elif step < FORWARD_N + REVEAL_N + BACKWARD_N:
        phase = "backward"
        progress = step - FORWARD_N - REVEAL_N
        banner = "backward pass: gradients flow right to left"
        banner_color = BACKWARD_COLOR
    else:
        phase, progress = "final", 0
        banner = "backward pass: gradients flow right to left"
        banner_color = BACKWARD_COLOR

    fig, ax = plt.subplots(figsize=(9.6, 7.4), dpi=120)
    fig.subplots_adjust(top=0.90, bottom=0.03, left=0.19, right=0.97)

    for i, row_top in enumerate(ROW_TOPS):
        draw_row(ax, row_top, i, phase, progress)

    ax.set_xlim(-2.0, 7.6)
    ax.set_ylim(-0.35, 7.1)
    ax.axis("off")

    ax.text(2.6, -0.2, LEGEND_TEXT, ha="center", va="center", fontsize=8.3,
             color=TEXT_COLOR, fontweight="bold")

    fig.suptitle("three transfer-learning strategies", fontsize=12.5, color=TEXT_COLOR,
                  fontweight="bold", y=0.975)
    fig.text(0.5, 0.925, banner, ha="center", fontsize=10, color=banner_color, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 7
HOLD_REVEAL = 20
HOLD_FINAL = 90
total_steps = FORWARD_N + REVEAL_N + BACKWARD_N + HOLD_N
for step in range(total_steps):
    is_final = step == total_steps - 1
    is_reveal = FORWARD_N <= step < FORWARD_N + REVEAL_N
    hold = HOLD_FINAL if is_final else (HOLD_REVEAL if is_reveal else HOLD_STEP)
    frames += [render(step)] * hold

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=60,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
