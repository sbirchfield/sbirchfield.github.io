"""Generate causal_attention_generation_demo.gif: an animated companion to
the lesson 45 notebook's "Causal masking" section. The notebook's own
causal-mask heatmap is static and shows the finished (T, T) triangle; this
animation shows *why* it's a triangle by tying it to generation order: a
short token sequence is produced one position at a time, and at each step
t, only the just-generated token's attention row is revealed -- always
exactly the weights over positions 0..t, never anything past it, because
position t+1 doesn't exist yet when position t is generated.

Also plants the seed for the forward reference to the KV cache (Lesson 53):
every row already generated keeps the same keys/values it had when it was
produced -- nothing about an old row's weights changes as new rows are
added, a fact this animation makes visually obvious (filled rows never
change) but does not name.

Run from anywhere:
    python make_causal_attention_generation_demo.py
Output:
    causal_attention_generation_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "causal_attention_generation_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"
GEN_FACE = "#d6e8d6"
PENDING_FACE = "#f0f0f0"
MASK_FACE = "#ffffff"

TOKENS = ["the", "cat", "sat", "on", "the", "mat"]
T = len(TOKENS)

rng = np.random.default_rng(45)
D = 8
Q = rng.normal(size=(T, D))
K = rng.normal(size=(T, D))
scores = Q @ K.T / np.sqrt(D)

weights = np.full((T, T), np.nan)
for t in range(T):
    row = scores[t, :t + 1]
    w = np.exp(row - row.max())
    w /= w.sum()
    weights[t, :t + 1] = w


def render(t):
    fig = plt.figure(figsize=(9.6, 5.2), dpi=120)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], left=0.06, right=0.97,
                            top=0.78, bottom=0.1, wspace=0.28)
    ax_seq = fig.add_subplot(gs[0, 0])
    ax_mat = fig.add_subplot(gs[0, 1])

    # Token sequence: generated so far (green), current (orange), not yet generated (grey).
    for j, tok in enumerate(TOKENS):
        if j < t:
            face, ecolor, lw = GEN_FACE, "#2e9e44", 1.2
        elif j == t:
            face, ecolor, lw = ACTIVE_COLOR, ACTIVE_COLOR, 2.4
        else:
            face, ecolor, lw = PENDING_FACE, "#aaaaaa", 1.0
        ax_seq.add_patch(Rectangle((0, T - 1 - j - 0.4), 1.6, 0.8, facecolor=face,
                                     edgecolor=ecolor, linewidth=lw, zorder=2))
        text_color = "white" if j == t else NEUTRAL_COLOR
        label = tok if j <= t else "?"
        ax_seq.text(0.8, T - 1 - j, label, ha="center", va="center", fontsize=11,
                     color=text_color, fontweight="bold" if j == t else "normal", zorder=3)
        ax_seq.text(-0.25, T - 1 - j, f"pos {j}", ha="right", va="center", fontsize=8.5, color="#777777")
    ax_seq.set_xlim(-1.3, 1.9)
    ax_seq.set_ylim(-0.7, T - 0.3)
    ax_seq.axis("off")
    ax_seq.set_title("generating left to right", fontsize=11)

    # Attention-weight matrix: only rows 0..t are revealed this frame, rest masked white.
    disp_vals = np.full((T, T), np.nan)
    disp_vals[:t + 1] = weights[:t + 1]
    disp = np.ma.masked_invalid(disp_vals)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(MASK_FACE)
    vmax = np.nanmax(weights)
    ax_mat.imshow(disp, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
    ax_mat.set_xticks(range(T)); ax_mat.set_xticklabels(range(T), fontsize=8.5)
    ax_mat.set_yticks(range(T)); ax_mat.set_yticklabels(range(T), fontsize=8.5)
    ax_mat.set_xlabel("key position (attended to)", fontsize=9.5)
    ax_mat.set_ylabel("query position (generated at this step)", fontsize=9.5)
    ax_mat.add_patch(Rectangle((-0.5, t - 0.5), T, 1, fill=False, edgecolor=ACTIVE_COLOR, linewidth=2.6, zorder=5))
    # future rows: dim grey wash so the "not yet generated" rows read as visibly withheld
    if t < T - 1:
        ax_mat.add_patch(Rectangle((-0.5, t + 0.5), T, T - t - 1.5, facecolor="#dddddd", alpha=0.6, zorder=4))
    ax_mat.set_title("causal attention weights, revealed row by row", fontsize=10.5)

    fig.suptitle(f"step {t+1}/{T}: token \"{TOKENS[t]}\" attends only to positions 0..{t}",
                 fontsize=12.5, color=NEUTRAL_COLOR, fontweight="bold", y=0.95)
    fig.text(0.5, 0.875, "a position's weights, once generated, never change as later positions are added",
              ha="center", fontsize=9.5, color="#666666", style="italic")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 45
HOLD_END = 90
for t in range(T):
    frames += [render(t)] * (HOLD_END if t == T - 1 else HOLD_STEP)

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=60,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
