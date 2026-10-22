"""Generate boundary_modes.png: a static side-by-side comparison of the
three border padding modes used by cv2.copyMakeBorder (CONSTANT, REPLICATE,
REFLECT101), with an arrow from every padding cell back to the interior
pixel it copies its value from (no arrow for CONSTANT, which is always
zero).

Uses the same 4x4 array as the lesson 10 notebook's boundary handling cell.

Run from anywhere:
    python make_boundary_static.py
Output:
    boundary_modes.png  (written next to this script)
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(OUT_DIR, "boundary_modes.png")

# Same array as the lesson 10 notebook's boundary handling example.
small = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
SIZE = 4
PAD = 1
N = SIZE + 2 * PAD  # 6

INTERIOR_FACE = "#ffffff"
BORDER_FACE = "#fdf0dc"
ARROW_COLOR = "#e07b00"
TEXT_COLOR = "#222222"


def replicate_idx(i, size):
    return min(max(i, 0), size - 1)


def reflect101_idx(i, size):
    if i < 0:
        return -i
    if i >= size:
        return 2 * (size - 1) - i
    return i


MODES = [
    ("CONSTANT (zero padding)", None),
    ("REPLICATE", replicate_idx),
    ("REFLECT101", reflect101_idx),
]

EXPLANATION = {
    "CONSTANT (zero padding)": "always 0",
    "REPLICATE": "copies the nearest edge pixel",
    "REFLECT101": "mirrors across the edge, skipping it",
}


def padded_values_and_sources(idx_fn):
    """Returns a (N, N) value grid and, for border cells, a dict mapping
    (r, c) -> source (r, c) in the same padded coordinate system."""
    values = np.zeros((N, N))
    sources = {}
    for r in range(N):
        for c in range(N):
            is_border = r < PAD or r >= PAD + SIZE or c < PAD or c >= PAD + SIZE
            if not is_border:
                values[r, c] = small[r - PAD, c - PAD]
                continue
            if idx_fn is None:
                values[r, c] = 0.0
            else:
                sr = idx_fn(r - PAD, SIZE)
                sc = idx_fn(c - PAD, SIZE)
                values[r, c] = small[sr, sc]
                sources[(r, c)] = (sr + PAD, sc + PAD)
    return values, sources


def draw_panel(ax, mode_i):
    name, idx_fn = MODES[mode_i]
    values, sources = padded_values_and_sources(idx_fn)

    for r in range(N):
        for c in range(N):
            is_border = r < PAD or r >= PAD + SIZE or c < PAD or c >= PAD + SIZE
            face = BORDER_FACE if is_border else INTERIOR_FACE
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=face,
                                     edgecolor="#999999", linewidth=0.8, zorder=0))
            ax.text(c, r, f"{values[r, c]:g}", ha="center", va="center",
                     fontsize=10, color=TEXT_COLOR, zorder=2)

    ax.add_patch(Rectangle((PAD - 0.5, PAD - 0.5), SIZE, SIZE, fill=False,
                             edgecolor="#555555", linewidth=2.2, zorder=1))

    # REFLECT101's source is exactly as far past the edge as the padding
    # cell is before it, so a straight arrow passes directly through the
    # (unused) edge pixel -- curve it instead, so the skip is visible.
    curve = name == "REFLECT101"
    for (r, c), (sr, sc) in sources.items():
        if (r, c) == (sr, sc):
            continue
        connectionstyle = "arc3,rad=0.3" if curve else None
        ax.annotate("", xy=(sc, sr), xytext=(c, r),
                     arrowprops=dict(arrowstyle="->", color=ARROW_COLOR,
                                      lw=1.0, alpha=0.75, shrinkA=7, shrinkB=7,
                                      connectionstyle=connectionstyle),
                     zorder=3)

    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(N - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_autoscale_on(False)
    ax.axis("off")

    ax.set_title(name, fontsize=14, fontweight="bold", color="#222222", pad=10)
    ax.text(0.5, -0.06, EXPLANATION[name], transform=ax.transAxes,
             ha="center", va="top", fontsize=9.5, color="#444444")


fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), dpi=140)
fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.08, wspace=0.12)

for i, ax in enumerate(axes):
    draw_panel(ax, i)

fig.savefig(OUT_PNG)
plt.close(fig)
print(f"Saved {OUT_PNG}")
