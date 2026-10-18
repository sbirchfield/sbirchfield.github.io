"""Generate conv_depth_demo.gif: an animated companion to the lesson 34
notebook's "A conv layer kernel is 3D" section, in the style of the
solid-cuboid CNN diagrams common in the literature (e.g. towardsdatascience
"CNN architecture explained" / "types of convolution kernels"): the input
is drawn as a true-looking 3D volume (10x10x3, three stacked channel
slabs, colored red/green/blue), and a solid 3x3x3 kernel cuboid slides
across it, one output position at a time. No per-cell arithmetic is shown
(that's covered elsewhere); instead, a small panel ties the geometric
picture to the algebra: the current window and the kernel are drawn as
flattened, color-blocked bars feeding a single dot product, landing on the
matching cell of the output feature map.

Reuses the same kernel + bias as the earlier per-channel demo (verified
values: vertical-edge / uniform / center-weighted slices, bias=0.2) for
continuity, applied here to a larger, purely illustrative striped volume
(no claim of real data -- parallel to make_conv_projection_demo.py's
synthetic step-edge image).

Run from anywhere:
    python make_conv_depth_demo.py
Output:
    conv_depth_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "conv_depth_demo.gif")

CH_COLORS = ["#d62728", "#2e9e44", "#1f6fb4"]  # channel 0 = red, 1 = green, 2 = blue
HIGHLIGHT_COLOR = "#f2c200"
TEXT_COLOR = "#222222"

H, W, C = 10, 10, 3
KH, KW = 3, 3
OH, OW = H - KH + 1, W - KW + 1

# a purely illustrative striped volume (no claim to real data, like the
# step-edge image used in make_conv_projection_demo.py)
rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
img = np.zeros((C, H, W), dtype=np.float64)
img[0] = ((cols // 2 + rows % 2) % 2)
img[1] = ((rows // 2 + cols % 2) % 2)
img[2] = (((rows + cols) // 2) % 2)

kernel = np.array([
    [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
    [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
    [[1, 1, 1], [1, 2, 1], [1, 1, 1]],
], dtype=np.float64)
bias = 0.2

output = np.zeros((OH, OW), dtype=np.float64)
for orow in range(OH):
    for ocol in range(OW):
        patch = img[:, orow:orow + KH, ocol:ocol + KW]
        output[orow, ocol] = float(np.sum(patch * kernel)) + bias

POSITIONS = [(r, c) for r in reversed(range(OH)) for c in range(OW)]  # back row to front row

ISO_COS = np.cos(np.radians(30))
ISO_SIN = np.sin(np.radians(30))


def iso_pt(u, v, w):
    return (u - v) * ISO_COS, (u + v) * ISO_SIN + w


def shade(color, factor, lighten=True):
    rgb = np.array(mcolors.to_rgb(color))
    rgb = rgb + (1 - rgb) * factor if lighten else rgb * (1 - factor)
    return tuple(np.clip(rgb, 0, 1))


def draw_iso_box(ax, u0, v0, w0, du, dv, dw, color, alpha=1.0, zorder=0,
                  edgecolor="#333333", lw=0.6):
    top = [(u0, v0, w0 + dw), (u0 + du, v0, w0 + dw), (u0 + du, v0 + dv, w0 + dw), (u0, v0 + dv, w0 + dw)]
    left = [(u0, v0, w0), (u0, v0 + dv, w0), (u0, v0 + dv, w0 + dw), (u0, v0, w0 + dw)]
    front = [(u0, v0, w0), (u0 + du, v0, w0), (u0 + du, v0, w0 + dw), (u0, v0, w0 + dw)]
    faces = [(front, shade(color, 0.35, lighten=False)),
             (left, color),
             (top, shade(color, 0.35, lighten=True))]
    for face, c in faces:
        pts = [iso_pt(*p) for p in face]
        ax.add_patch(Polygon(pts, closed=True, facecolor=c, edgecolor=edgecolor,
                               linewidth=lw, alpha=alpha, zorder=zorder))


def draw_top_grid(ax, u0, v0, w0, du, dv, n_u, n_v, color="#ffffff", lw=0.5, alpha=0.6, zorder=0):
    for i in range(n_u + 1):
        u = u0 + i * du / n_u
        p0, p1 = iso_pt(u, v0, w0), iso_pt(u, v0 + dv, w0)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=lw, alpha=alpha, zorder=zorder)
    for j in range(n_v + 1):
        v = v0 + j * dv / n_v
        p0, p1 = iso_pt(u0, v, w0), iso_pt(u0 + du, v, w0)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=lw, alpha=alpha, zorder=zorder)


STACK_ORDER = [2, 1, 0]  # bottom-to-top channel indices: channel 0 (red) ends up on top


def draw_volume(ax, row, col):
    for w0, ch in enumerate(STACK_ORDER):
        draw_iso_box(ax, 0, 0, w0, W, H, 1, CH_COLORS[ch], alpha=0.5, zorder=w0)
    draw_top_grid(ax, 0, 0, 3, W, H, W, H, color="#ffffff", lw=0.4, alpha=0.5, zorder=3)
    draw_iso_box(ax, col, row, 0, KW, KH, 3, HIGHLIGHT_COLOR, alpha=0.55, zorder=10,
                 edgecolor="#5a4600", lw=1.8)
    draw_top_grid(ax, col, row, 3, KW, KH, KW, KH, color="#5a4600", lw=0.8, alpha=0.9, zorder=11)


def render(step):
    orow, ocol = POSITIONS[step]
    fig = plt.figure(figsize=(12.4, 7.2), dpi=120)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.4, 1.0], width_ratios=[2.1, 1.0],
                            hspace=0.26, wspace=0.18, top=0.86, bottom=0.08, left=0.03, right=0.97)
    ax_vol = fig.add_subplot(gs[0, 0])
    ax_out = fig.add_subplot(gs[0, 1])
    ax_flat = fig.add_subplot(gs[1, :])

    draw_volume(ax_vol, orow, ocol)
    ax_vol.set_xlim(-(H + 1) * ISO_COS - 1, (W + 1) * ISO_COS + 4)
    ax_vol.set_ylim(-1, (W + H) * ISO_SIN + 3 + 1)
    ax_vol.set_aspect("equal")
    ax_vol.axis("off")
    for w0, ch in enumerate(STACK_ORDER):
        lx, ly = iso_pt(W + 0.3, 0, w0 + 0.5)
        ax_vol.text(lx, ly, f"channel {ch}", fontsize=8.5, color=CH_COLORS[ch],
                     fontweight="bold", ha="left", va="center")

    disp = np.full((OH, OW), np.nan)
    for r, c in POSITIONS[:step]:
        disp[r, c] = output[r, c]
    vmin, vmax = output.min(), output.max()
    masked = np.ma.masked_invalid(disp)
    cmap = plt.cm.Greys
    cmap.set_bad("#ffffff")
    ax_out.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower",
                   extent=(-0.5, OW - 0.5, -0.5, OH - 0.5))
    ax_out.set_xticks(np.arange(-0.5, OW, 1), minor=True)
    ax_out.set_yticks(np.arange(-0.5, OH, 1), minor=True)
    ax_out.grid(which="minor", color="#bbbbbb", linewidth=0.5)
    ax_out.set_xticks([])
    ax_out.set_yticks([])
    ax_out.add_patch(Rectangle((ocol - 0.5, orow - 0.5), 1, 1, facecolor="none",
                                  edgecolor=HIGHLIGHT_COLOR, linewidth=2.6, zorder=5))
    ax_out.set_xlim(-0.5, OW - 0.5)
    ax_out.set_ylim(-0.5, OH - 0.5)
    ax_out.set_title("output feature map (8×8)", fontsize=10.5, color=TEXT_COLOR)

    ax_flat.axis("off")
    ax_flat.set_xlim(-7, 41)
    ax_flat.set_ylim(0, 3)

    patch = img[:, orow:orow + KH, ocol:ocol + KW]

    def draw_bar(y, label, values):
        vflat = values.flatten()
        vlo, vhi = vflat.min(), vflat.max()
        vspan = vhi - vlo if vhi > vlo else 1.0
        for i in range(27):
            color = CH_COLORS[i // 9]
            t = (vflat[i] - vlo) / vspan
            alpha = 0.22 + 0.63 * t
            ax_flat.add_patch(Rectangle((i, y), 1, 0.9, facecolor=color, alpha=alpha,
                                           edgecolor="white", linewidth=0.6))
        ax_flat.text(-0.8, y + 0.45, label, ha="right", va="center", fontsize=10.5,
                       color=TEXT_COLOR, fontweight="bold")

    draw_bar(1.7, "kernel w =", kernel)
    draw_bar(0.5, "window x =", patch)
    ax_flat.annotate("", xy=(30.3, 1.2), xytext=(27.4, 1.2),
                       arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=1.8))
    ax_flat.text(31.0, 1.2, "y = wᵀx + b =", ha="left", va="center", fontsize=12,
                   color=TEXT_COLOR, fontweight="bold")

    val = output[orow, ocol]
    out_color = cmap(0.5 if vmax <= vmin else (val - vmin) / (vmax - vmin))
    ax_flat.add_patch(Rectangle((37.0, 0.85), 1.3, 0.75, facecolor=out_color,
                                   edgecolor="#333333", linewidth=1.3, zorder=5))

    ax_flat.set_title("                              Flattened:", loc="left", fontsize=11.5, color=TEXT_COLOR, fontweight="bold")

    fig.suptitle("Convolving a 10x10 RGB image with a 3×3×3 kernel", fontsize=13, color=TEXT_COLOR, y=0.95)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 3
HOLD_END = 90
n_steps = OH * OW
for step in range(n_steps):
    frames += [render(step)] * HOLD_STEP
frames += [render(n_steps - 1)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=55,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames); output range=[{output.min():.2f}, {output.max():.2f}]")
