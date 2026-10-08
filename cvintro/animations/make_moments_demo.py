"""Generate moments_demo.gif: an animated, step-by-step walkthrough of computing
raw moments (m00, m10, m01) and the centroid on a small binary image, by hand.

Run from anywhere:
    python make_moments_demo.py
Output:
    moments_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "moments_demo.gif")

# A small binary "blob" (an L-shape), bigger than the 3x3 example in the lesson
# but still small enough to read every pixel's coordinates at a glance.
im = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=np.uint8)

H, W = im.shape
ys, xs = np.nonzero(im)
order = np.lexsort((xs, ys))  # visit in reading order (row by row)
ys, xs = ys[order], xs[order]

FG = "#ffffff"       # unvisited foreground pixel: white
BG = "#1f1f1f"       # background pixel: near-black
DOT_COLOR = "#e54040"    # dot marking a pixel that has been summed: red
STAR_COLOR = "#1f6fd6"   # centroid marker: blue
DOT_RADIUS = 0.22    # small circle, well short of filling the pixel cell

# (subscript, term shown in the sum, per-pixel contribution)
M_SPECS = [
    ("00", "1", lambda x, y: 1),
    ("10", "x", lambda x, y: x),
    ("01", "y", lambda x, y: y),
]


def render(step, hold_centroid=False):
    """step: -1 for the blank intro (nothing summed yet), an index into
    (xs, ys) for the pixel just added to the running sums, or len(xs) once
    everything has been summed (centroid phase)."""
    rgb = np.zeros((H, W, 3))
    rgb[im == 1] = 1.0
    # A dot appearing and its contribution to the sums happen in the same
    # frame -- "included" counts the pixel at `step` itself, not just the
    # ones before it.
    included = min(step + 1, len(xs)) if step >= 0 else 0
    is_stepping = 0 <= step < len(xs)

    # Fixed figure/axes layout every frame (no tight_layout, no autoscale)
    # so the grid never shifts or "zooms" between frames.
    fig, (ax_img, ax_text) = plt.subplots(
        1, 2, figsize=(8.4, 5.2), dpi=120,
        gridspec_kw={"width_ratios": [1.0, 0.85]},
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.08, wspace=0.05)

    ax_img.imshow(rgb, interpolation="nearest", cmap="gray", vmin=0, vmax=1)
    ax_img.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax_img.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax_img.grid(which="minor", color="#888888", linewidth=0.8)
    ax_img.set_xticks(range(W))
    ax_img.set_yticks(range(H))
    ax_img.tick_params(length=0)
    ax_img.set_xlim(-0.5, W - 0.5)
    ax_img.set_ylim(H - 0.5, -0.5)
    ax_img.set_aspect("equal")
    ax_img.set_autoscale_on(False)

    for i in range(included):
        ax_img.add_patch(plt.Circle((xs[i], ys[i]), DOT_RADIUS,
                                     facecolor=DOT_COLOR, edgecolor="none", zorder=3))

    if step < 0:
        title = "binary image"
    elif is_stepping:
        x, y = xs[step], ys[step]
        title = f"visiting pixel (x={x}, y={y})"
    else:
        title = "all foreground pixels visited"

    ax_img.set_title(title, fontsize=13)

    # Text panel: fixed line positions regardless of which lines are shown,
    # so the layout never resizes/shifts between frames.
    ax_text.axis("off")
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)

    totals = {}
    line_specs = []
    for (sub, term, contrib), y in zip(M_SPECS, (0.82, 0.67, 0.52)):
        prev_total = sum(contrib(xs[i], ys[i]) for i in range(included - 1)) if is_stepping else None
        total = sum(contrib(xs[i], ys[i]) for i in range(included))
        totals[sub] = total
        prefix = f"$m_{{{sub}}}$ = Σ{term}"
        if is_stepping:
            this_val = contrib(xs[step], ys[step])
            text = f"{prefix} = {prev_total} + {this_val} = {total}"
        else:
            text = f"{prefix} = {total}"
        line_specs.append((y, text))

    # Centroid text and both stars reveal together, not before.
    show_centroid = included > 0 and step >= len(xs) and hold_centroid
    cx = cy = None
    if show_centroid:
        cx, cy = totals["10"] / totals["00"], totals["01"] / totals["00"]
        line_specs.append((0.33, f"centroid = ({cx:.2f}, {cy:.2f})"))

    for y, text in line_specs:
        ax_text.text(0.14, y, text, va="center", ha="left",
                      fontsize=13, family="monospace", transform=ax_text.transAxes)

    if show_centroid and hold_centroid:
        # Same blue star as on the image, placed right next to the word
        # "centroid" so the connection between the two is obvious.
        ax_text.plot(0.095, 0.33, marker="*", markersize=16, color=STAR_COLOR,
                      markeredgecolor="white", markeredgewidth=1.0,
                      transform=ax_text.transAxes, clip_on=False, zorder=5)

    if hold_centroid and show_centroid:
        ax_img.plot(cx, cy, marker="*", markersize=22, color=STAR_COLOR,
                     markeredgecolor="white", markeredgewidth=1.2, zorder=5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_INTRO = 6
HOLD_STEP = 16
HOLD_END = 20

# Intro: show the blank binary image before any pixel is visited.
intro = render(-1)
frames += [intro] * HOLD_INTRO

# Step through each foreground pixel.
for step in range(len(xs)):
    frames += [render(step)] * HOLD_STEP

# Final: all pixels visited, sums complete, then centroid text and both
# stars appear together and hold for twice as long as a normal step.
frames += [render(len(xs))] * HOLD_STEP
frames += [render(len(xs), hold_centroid=True)] * (HOLD_END * 2)

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=90,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
