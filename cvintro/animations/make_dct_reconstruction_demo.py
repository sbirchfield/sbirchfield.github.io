"""Generate dct_reconstruction_demo.gif: an animated version of the lesson 17
notebook's "Quantization: throwing away the coefficients that matter least"
cell. Takes the same 8x8 block (straddling the rectangle's sharp edge in
`photo_like`) and its DCT, then reconstructs it from an inverse DCT while
keeping more and more coefficients in zigzag order (low frequency first),
showing the block sharpen progressively from a flat DC-only approximation to
the exact original.

Run from anywhere:
    python make_dct_reconstruction_demo.py
Output:
    dct_reconstruction_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "dct_reconstruction_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"

# Same synthetic "photo_like" image and block as the notebook cell.
rng = np.random.default_rng(0)
photo_like = np.zeros((100, 100), dtype=np.uint8)
cv2.rectangle(photo_like, (10, 10), (90, 90), 200, -1)
cv2.circle(photo_like, (50, 50), 25, 120, -1)
photo_like = np.clip(photo_like.astype(np.float64) + rng.normal(0, 5, photo_like.shape), 0, 255).astype(np.uint8)

block = photo_like[6:14, 6:14].astype(np.float64)
dct_block = cv2.dct(block)


def zigzag_order(n=8):
    """Standard JPEG zigzag traversal order of an nxn coefficient grid,
    visiting low frequencies (top-left) before high frequencies (bottom-right)."""
    coords = []
    for s in range(2 * n - 1):
        diag = [(i, s - i) for i in range(n) if 0 <= s - i < n]
        if s % 2 == 0:
            diag.reverse()
        coords.extend(diag)
    return coords


ZIGZAG = zigzag_order(8)


def render(num_kept, just_added):
    mask = np.zeros_like(dct_block)
    for (r, c) in ZIGZAG[:num_kept]:
        mask[r, c] = 1
    kept = dct_block * mask
    reconstructed = cv2.idct(kept)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), dpi=120,
                              gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]})
    fig.subplots_adjust(top=0.78, bottom=0.1, wspace=0.3)
    ax_orig, ax_mask, ax_recon = axes

    ax_orig.imshow(block, cmap="gray", vmin=block.min(), vmax=block.max())
    ax_orig.set_title("original block", fontsize=10.5)
    ax_orig.set_xticks([])
    ax_orig.set_yticks([])

    ax_mask.imshow(np.zeros((8, 8)), cmap="gray", vmin=0, vmax=1)
    for r in range(8):
        for c in range(8):
            if mask[r, c]:
                color = ACTIVE_COLOR if (r, c) == ZIGZAG[num_kept - 1] else "#bbbbbb"
                ax_mask.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color,
                                             edgecolor="#555555", linewidth=0.5))
    ax_mask.set_xlim(-0.5, 7.5)
    ax_mask.set_ylim(7.5, -0.5)
    ax_mask.set_title("coefficients kept\n(zigzag order)", fontsize=10.5)
    ax_mask.set_xticks([])
    ax_mask.set_yticks([])

    ax_recon.imshow(reconstructed, cmap="gray", vmin=block.min(), vmax=block.max())
    ax_recon.set_title(f"reconstruction\n{num_kept}/64 coeffs kept", fontsize=10.5,
                        color=MATCH_COLOR if num_kept == 64 else NEUTRAL_COLOR)
    ax_recon.set_xticks([])
    ax_recon.set_yticks([])

    fig.suptitle("DCT reconstruction: more coefficients, sharper block", fontsize=12.5,
                 color=NEUTRAL_COLOR, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_START = 14
HOLD_STEP = 3
HOLD_END = 40

frames += [render(1, True)] * HOLD_START
for num_kept in range(2, 65):
    frames += [render(num_kept, True)] * HOLD_STEP
frames += [render(64, True)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=60,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
