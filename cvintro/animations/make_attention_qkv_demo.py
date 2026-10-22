"""Generate attention_qkv_demo.gif: an animated companion to the lesson 45
notebook's `attention(Q, K, V)` formula and `attention()` function. Computes
one output row at a time, for a small Q, K, V (T=5 positions, D=6
dimensions), making concrete what the formula abstracts away: for each
query row i, every key row gets a similarity score, softmax turns those
scores into weights that sum to 1, and the output row is nothing but a
weighted average of V's rows using those weights -- stepping through
exactly the same three stages (score, softmax, weighted sum) the notebook's
`attention()` function computes in one call, now spread out and labeled
with every tensor's shape.

Uses its own small fixed Q, K, V (not the notebook's own random tensors --
same T, same idea, chosen only for a clean, legible grid).

Run from anywhere:
    python make_attention_qkv_demo.py
Output:
    attention_qkv_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "attention_qkv_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"
PENDING_FACE = "#f7f7f7"

rng = np.random.default_rng(45)
T, D = 5, 6
Q = rng.normal(size=(T, D))
K = rng.normal(size=(T, D))
V = rng.normal(size=(T, D))

scores_full = Q @ K.T / np.sqrt(D)          # (T, T)
weights_full = np.exp(scores_full - scores_full.max(axis=1, keepdims=True))
weights_full /= weights_full.sum(axis=1, keepdims=True)
output_full = weights_full @ V              # (T, D)

VAL_CMAP = "RdBu_r"
VAL_CLIP = 2.5


def mat_panel(ax, M, title, dim_label, row_highlight=None, row_overlay_alpha=None):
    ax.imshow(M, cmap=VAL_CMAP, vmin=-VAL_CLIP, vmax=VAL_CLIP, aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color("#999999")
    if row_highlight is not None:
        n_rows, n_cols = M.shape
        ax.add_patch(Rectangle((-0.5, row_highlight - 0.5), n_cols, 1, fill=False,
                                 edgecolor=ACTIVE_COLOR, linewidth=2.8, zorder=5))
    if row_overlay_alpha is not None:
        n_rows, n_cols = M.shape
        for r, a in enumerate(row_overlay_alpha):
            if a > 0.02:
                ax.add_patch(Rectangle((-0.5, r - 0.5), n_cols, 1, facecolor=MATCH_COLOR,
                                         edgecolor="none", alpha=min(a, 0.85), zorder=4))
    ax.set_title(f"{title}\n{dim_label}", fontsize=10)


def weights_panel(ax, values, title, is_weights):
    cmap = "viridis" if is_weights else "RdBu_r"
    vmin, vmax = (0, max(values.max(), 1e-6)) if is_weights else (-VAL_CLIP, VAL_CLIP)
    ax.imshow(values.reshape(1, -1), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_yticks([])
    ax.set_xticks(range(T)); ax.set_xticklabels([f"k{j}" for j in range(T)], fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color("#999999")
    for j, v in enumerate(values):
        ax.text(j, 0, f"{v:.2f}", ha="center", va="center", fontsize=8,
                 color="white" if is_weights and v > values.max() * 0.4 else NEUTRAL_COLOR)
    ax.set_title(title, fontsize=9.5)


def render(i, sub):
    # sub: 0 = highlight query row i, 1 = scores computed, 2 = softmax + weighted sum -> output row i
    fig = plt.figure(figsize=(9.6, 6.2), dpi=120)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1.0], width_ratios=[1, 1, 1],
                            left=0.05, right=0.97, top=0.80, bottom=0.07, hspace=0.55, wspace=0.18)

    ax_Q = fig.add_subplot(gs[0, 0])
    ax_K = fig.add_subplot(gs[0, 1])
    ax_V = fig.add_subplot(gs[0, 2])
    ax_bar = fig.add_subplot(gs[1, 0:2])
    ax_out = fig.add_subplot(gs[1, 2])

    mat_panel(ax_Q, Q, "Q", f"(T={T}, D={D})", row_highlight=i)
    row_overlay = weights_full[i] if sub == 2 else None
    mat_panel(ax_K, K, "K", f"(T={T}, D={D})")
    mat_panel(ax_V, V, "V", f"(T={T}, D={D})", row_overlay_alpha=row_overlay)

    if sub == 0:
        ax_bar.axis("off")
        ax_bar.text(0.5, 0.5, f"query row Q[{i}]  ->  compare against every key row",
                     ha="center", va="center", fontsize=11, color=ACTIVE_COLOR, transform=ax_bar.transAxes)
        stage_title = f"step {i+1}/{T}: pick query row i={i}"
    elif sub == 1:
        weights_panel(ax_bar, scores_full[i], f"scores_i = Q[{i}] K" + r"$^\top$" + f" / sqrt(D)   shape (1, {T})", is_weights=False)
        stage_title = f"step {i+1}/{T}: raw similarity scores against all {T} keys"
    else:
        weights_panel(ax_bar, weights_full[i], f"weights_i = softmax(scores_i)   shape (1, {T}), sums to 1", is_weights=True)
        stage_title = f"step {i+1}/{T}: softmax weights -> weighted average of V's rows"

    # Output matrix: rows < i filled in from previous steps; row i filled only once sub==2.
    out_display = np.zeros((T, D))
    out_display[:] = np.nan
    filled_rows = i if sub < 2 else i + 1
    out_display[:filled_rows] = output_full[:filled_rows]
    out_masked = np.ma.masked_invalid(out_display)
    cmap = plt.get_cmap(VAL_CMAP).copy()
    cmap.set_bad(PENDING_FACE)
    ax_out.imshow(out_masked, cmap=cmap, vmin=-VAL_CLIP, vmax=VAL_CLIP, aspect="auto")
    ax_out.set_xticks([]); ax_out.set_yticks([])
    for spine in ax_out.spines.values():
        spine.set_visible(True); spine.set_color("#999999")
    if sub == 2:
        ax_out.add_patch(Rectangle((-0.5, i - 0.5), D, 1, fill=False, edgecolor=MATCH_COLOR, linewidth=2.8, zorder=5))
    elif sub in (0, 1) and i < T:
        ax_out.add_patch(Rectangle((-0.5, i - 0.5), D, 1, fill=False, edgecolor=ACTIVE_COLOR,
                                     linewidth=1.6, linestyle="--", zorder=5))
    ax_out.set_title(f"Output\n(T={T}, D={D}) -- filled row by row", fontsize=10)

    fig.suptitle(stage_title, fontsize=13, color=NEUTRAL_COLOR, fontweight="bold", y=0.93)
    fig.text(0.5, 0.855, r"Attention(Q,K,V) = softmax(QK$^\top$/$\sqrt{d_k}$)V $\,-\,$ one row of the formula, made concrete",
              ha="center", fontsize=9.5, color="#666666")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 16
HOLD_END = 70
for i in range(T):
    for sub in (0, 1, 2):
        frames += [render(i, sub)] * HOLD_STEP
frames += [frames[-1]] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=65,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
