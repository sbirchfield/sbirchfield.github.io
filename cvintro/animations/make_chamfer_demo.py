"""Generate chamfer_demo.gif: an animated walkthrough of the two-pass 3-4
chamfer distance transform from the lesson 7 notebook's "Chamfer distance
transform" section: a forward raster pass (top-left -> bottom-right) followed
by a backward raster pass (bottom-right -> top-left), each cell taking the
minimum of its causal neighbors' distance plus a step cost (3 for
axis-aligned, 4 for diagonal).

Uses a small synthetic grid (not the notebook's actual image) so individual
cells and their numeric values are large enough to read.

Run from anywhere:
    python make_chamfer_demo.py
Output:
    chamfer_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "chamfer_demo.gif")

H, W = 6, 8
SEED_CELLS = [(2, 2), (2, 3), (3, 2), (3, 3)]  # a small 2x2 seed blob
A = 3  # axis-aligned step cost
B = 4  # diagonal step cost
INF = 10_000

ACTIVE_COLOR = "#e07b00"
COST_A_COLOR = "#1f6f4a"  # axis-aligned step cost (3)
COST_B_COLOR = "#7a3b96"  # diagonal step cost (4)
SEED_COLOR = "#c21807"
MATCH_COLOR = "#2e9e44"


def cost_color(w):
    return COST_A_COLOR if w == A else COST_B_COLOR


FORWARD_OFFSETS = [(-1, -1, B), (-1, 0, A), (-1, 1, B), (0, -1, A)]
BACKWARD_OFFSETS = [(1, 1, B), (1, 0, A), (1, -1, B), (0, 1, A)]


def fmt(v):
    return "∞" if v >= INF else str(int(v))


def fmt_pad(v, width=2):
    return fmt(v).rjust(width)


def run_pass(dist, offsets, order):
    """Yield (r, c, candidates, before, after) for each cell visited in
    `order`, where candidates is a fixed-length list (always len(offsets))
    of (dr, dc, weight, neighbor_value_or_None, valid) -- keeping a slot for
    every offset, even off-grid ones, so the formula's layout never changes
    shape from one step to the next."""
    for r, c in order:
        candidates = []
        for dr, dc, w in offsets:
            nr, nc = r + dr, c + dc
            valid = 0 <= nr < H and 0 <= nc < W
            candidates.append((dr, dc, w, dist[nr, nc] if valid else None, valid))
        before = dist[r, c]
        best = before
        for dr, dc, w, nv, valid in candidates:
            if valid:
                best = min(best, nv + w)
        dist[r, c] = best
        yield r, c, candidates, before, best


dist = np.full((H, W), INF, dtype=int)
for sr, sc in SEED_CELLS:
    dist[sr, sc] = 0

forward_order = [(r, c) for r in range(H) for c in range(W)]
backward_order = [(r, c) for r in range(H - 1, -1, -1) for c in range(W - 1, -1, -1)]

steps = []  # (dist_snapshot, pass_name, (r, c), candidates, before, after)
steps.append((dist.copy(), None, None, [], None, None))
for r, c, candidates, before, after in run_pass(dist, FORWARD_OFFSETS, forward_order):
    steps.append((dist.copy(), "forward", (r, c), candidates, before, after))
for r, c, candidates, before, after in run_pass(dist, BACKWARD_OFFSETS, backward_order):
    steps.append((dist.copy(), "backward", (r, c), candidates, before, after))

FINAL_DIST = steps[-1][0]
VMAX = int(FINAL_DIST[np.isfinite(FINAL_DIST) & (FINAL_DIST < INF)].max())
CMAP = mpl.colormaps["viridis"]


LIGHTEN = 0.55  # blend fraction toward white, so cell fills stay pale and arrows stand out


def cell_color(v):
    if v >= INF:
        return "#eeeeee"
    t = 0 if VMAX == 0 else min(v / VMAX, 1.0)
    r, g, b, _ = CMAP(t)
    r = r + (1 - r) * LIGHTEN
    g = g + (1 - g) * LIGHTEN
    b = b + (1 - b) * LIGHTEN
    return (r, g, b)


def draw_rich_text(fig, y, segments, fontsize=12, x0=0.17):
    """segments: list of (text, color, bold). Lays them out left-to-right
    starting at a fixed `x0`, using the renderer to measure each token's
    actual width so colors can vary within one line of text. Anchoring at a
    fixed start (rather than centering) keeps shared tokens like "min(" and
    "=" from jumping around as the surrounding text changes length."""
    texts = []
    for s, color, bold in segments:
        t = fig.text(0, y, s, fontsize=fontsize, color=color, family="monospace",
                     fontweight="bold" if bold else "normal", ha="left", va="center")
        texts.append(t)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths = [t.get_window_extent(renderer=renderer).width for t in texts]
    fig_width_px = fig.bbox.width
    x = x0
    for t, w in zip(texts, widths):
        t.set_position((x, y))
        x += w / fig_width_px


def draw_centered_text(fig, y, text, color, fontsize=11, bold=True):
    fig.text(0.5, y, text, fontsize=fontsize, color=color, ha="center", va="center",
               fontweight="bold" if bold else "normal")


def draw_rich_text_centered(fig, y, segments, fontsize=12, center_x=0.5):
    """Like draw_rich_text, but centers the whole line at `center_x` instead
    of anchoring its start -- used for the title, where the full line is
    short and static enough that centering doesn't cause noticeable jitter."""
    texts = []
    for s, color, bold in segments:
        t = fig.text(0, y, s, fontsize=fontsize, color=color,
                     fontweight="bold" if bold else "normal", ha="left", va="center")
        texts.append(t)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths = [t.get_window_extent(renderer=renderer).width for t in texts]
    fig_width_px = fig.bbox.width
    total_frac = sum(widths) / fig_width_px
    x = center_x - total_frac / 2
    for t, w in zip(texts, widths):
        t.set_position((x, y))
        x += w / fig_width_px


def render(step_i, final_hold=False):
    dist_snap, pass_name, active, candidates, before, after = steps[step_i]

    fig = plt.figure(figsize=(7.6, 5.6), dpi=120)
    gs = fig.add_gridspec(2, 1, height_ratios=[6.0, 1.0], top=0.84, bottom=0.08,
                            left=0.03, right=0.97, hspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax_legend = fig.add_subplot(gs[1])

    ax.set_aspect("equal")
    ax.axis("off")

    active = None if final_hold else active

    for r in range(H):
        for c in range(W):
            v = dist_snap[r, c]
            is_seed = (r, c) in SEED_CELLS
            is_active = active == (r, c)
            face = SEED_COLOR if is_seed else cell_color(v)
            edge = ACTIVE_COLOR if is_active else ("#333333" if is_seed else "#999999")
            lw = 2.2 if is_active else (1.6 if is_seed else 0.7)
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=face,
                                     edgecolor=edge, linewidth=lw, zorder=1))
            label = "0" if is_seed else fmt(v)
            text_color = "white" if is_seed else "#222222"
            fontsize = 12 if label == "∞" else 9
            ax.text(c, r, label, ha="center", va="center", fontsize=fontsize,
                     color=text_color, fontweight="bold" if is_active else "normal", zorder=2)

    if active is not None:
        ar, ac = active
        for dr, dc, w, nv, valid in candidates:
            if not valid:
                continue
            nr, nc = ar + dr, ac + dc
            ax.annotate("", xy=(ac, ar), xytext=(nc, nr),
                         arrowprops=dict(arrowstyle="->", color=cost_color(w), lw=1.8,
                                          shrinkA=10, shrinkB=10), zorder=3)

    ax.set_xlim(-0.6, W - 0.4)
    ax.set_ylim(H - 0.4, -0.6)

    # Legend: shows exactly the arrows currently drawn in the grid above
    # (same directions, same colors), split into the axis-aligned and
    # diagonal groups -- 0, 1, or 2 arrows per group, matching the active cell.
    if active is not None:
        axis_offs = [(dr, dc) for dr, dc, w, nv, valid in candidates if valid and w == A]
        diag_offs = [(dr, dc) for dr, dc, w, nv, valid in candidates if valid and w == B]
    else:
        axis_offs, diag_offs = [], []

    ax_legend.axis("off")
    ax_legend.set_xlim(0, 10)
    ax_legend.set_ylim(-0.6, 0.6)

    # The legend axes isn't square (wide x-range, short y-range), so a
    # direction vector normalized in data units ends up looking much shorter
    # vertically than horizontally. Measure the actual pixels-per-data-unit
    # along each axis and compensate, so every arrow has the same length
    # on screen regardless of its direction.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    p0 = ax_legend.transData.transform((0, 0))
    px = ax_legend.transData.transform((1, 0))
    py = ax_legend.transData.transform((0, 1))
    sx = abs(px[0] - p0[0])
    sy = abs(py[1] - p0[1])
    TARGET_PX = 26

    def draw_legend_arrows(anchor_x, offsets, color):
        for dr, dc in offsets:
            norm = (dr ** 2 + dc ** 2) ** 0.5
            ux, uy = dc / norm, -dr / norm
            start = (anchor_x + ux * TARGET_PX / sx, uy * TARGET_PX / sy)
            end = (anchor_x, 0)
            ax_legend.annotate("", xy=end, xytext=start,
                                 arrowprops=dict(arrowstyle="->", color=color, lw=2.2))

    draw_legend_arrows(0.7, axis_offs, COST_A_COLOR)
    ax_legend.text(1.3, 0, "+3 step cost (axis-aligned)", fontsize=9.5, color=COST_A_COLOR,
                    ha="left", va="center", fontweight="bold")
    draw_legend_arrows(6.1, diag_offs, COST_B_COLOR)
    ax_legend.text(6.6, 0, "+4 step cost (diagonal)", fontsize=9.5, color=COST_B_COLOR,
                    ha="left", va="center", fontweight="bold")

    draw_rich_text_centered(fig, 0.98, [
        ("(", "#222222", True), ("3", COST_A_COLOR, True), (",", "#222222", True),
        ("4", COST_B_COLOR, True), (")-chamfer distance transform", "#222222", True),
    ], fontsize=12.5)

    if final_hold:
        draw_centered_text(fig, 0.905, "complete", "#2e9e44")
        draw_centered_text(fig, 0.87, "divide by 3 for an approximate Euclidean distance", "#555555", fontsize=10, bold=False)
    elif pass_name is None:
        pass
    else:
        r, c = active
        shadow = "cast shadows down & right" if pass_name == "forward" else "cast shadows up & left"
        pass_label = "pass 1 of 2" if pass_name == "forward" else "pass 2 of 2"
        draw_centered_text(fig, 0.905, f"{pass_label} — {shadow}", ACTIVE_COLOR)
        segments = [(f"pixel (x={c}, y={r}):  min(", "#333333", False),
                     (fmt_pad(before), "#333333", False)]
        for dr, dc, w, nv, valid in candidates:
            segments.append((", ", "#333333", False))
            if valid:
                segments.append((fmt_pad(nv), "#333333", False))
                segments.append((f"+{w}", cost_color(w), False))
            else:
                segments.append(("    ", "#ffffff", False))
        segments.append((f") = {fmt_pad(after)}", "#333333", after < before))
        draw_rich_text(fig, 0.865, segments, fontsize=10.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_INTRO = 12
HOLD_CELL = 5
HOLD_PASS_END = 18
HOLD_DONE = 16
HOLD_END = 44

frames += [render(0)] * HOLD_INTRO
prev_pass = None
for i in range(1, len(steps)):
    _, pass_name, _, _, _, _ = steps[i]
    pass_just_ended = prev_pass is not None and pass_name != prev_pass
    prev_pass = pass_name
    if pass_just_ended:
        frames += [render(i - 1)] * HOLD_PASS_END
    frames += [render(i)] * HOLD_CELL
frames += [render(len(steps) - 1)] * HOLD_DONE
frames += [render(len(steps) - 1, final_hold=True)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=110,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames, {len(steps)} steps, grid {H}x{W}, vmax={VMAX})")
