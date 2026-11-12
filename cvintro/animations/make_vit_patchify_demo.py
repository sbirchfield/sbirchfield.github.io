"""Generate vit_patchify_demo.gif: an animated companion to the lesson 47
notebook's "Patchify: turning an image into a sequence" section. Carves a
16x16 single-channel image into its 16 non-overlapping 4x4 patches, one at
a time, in raster order -- highlighting the current patch on the image,
then flattening its 16 pixel values into a column that gets appended to a
growing "patch sequence" matrix below (patches as columns, pixels as rows --
matching vit_pipeline_demo.gif's P matrix, so the two animations agree).
Ends by holding the completed (16, 16) patch sequence, with a one-line
caption pointing at the pipeline animation for what happens to it next.

Deliberately does NOT show the patchify-equals-strided-conv equivalence
(that comparison lives only in the lesson's own code cell, id 0b2a28a0,
as a printed number) -- a Transformer never uses nn.Conv2d, so putting a
conv2d comparison in a diagram about the Transformer pipeline suggested
conv2d is part of that pipeline, which it isn't. Keeping this animation
to patchify alone, and leaving the linear embedding step to the pipeline
animation right after it, keeps each diagram's claim narrow and legible.

Reuses the notebook's own patchify() verbatim (cell id 0b2a28a0). The
image itself is this lesson's own make_image('plus', ...) shape (cell id
a0f686af), at size=16.

Run from anywhere:
    python make_vit_patchify_demo.py
Output:
    vit_patchify_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "vit_patchify_demo.gif")

ACTIVE_COLOR = "#e07b00"
DONE_COLOR = "#3a6fb0"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#555555"
PENDING_FACE = "#f7f7f7"


def make_image(shape_type, cx, cy, size=16):
    img = np.zeros((size, size), dtype=np.float32)
    if shape_type == "plus":
        img[cy - 1:cy + 2, cx - 3:cx + 4] = 1.0
        img[cy - 3:cy + 4, cx - 1:cx + 2] = 1.0
    else:
        yy, xx = np.mgrid[0:size, 0:size]
        img[((xx - cx) ** 2 + (yy - cy) ** 2) <= 9] = 1.0
    return img


def patchify(img, patch_size):
    B, C, H, W = img.shape
    P = patch_size
    patches = img.unfold(2, P, P).unfold(3, P, P)
    patches = patches.contiguous().view(B, C, -1, P, P).permute(0, 2, 1, 3, 4)
    return patches.reshape(B, -1, C * P * P)


SIZE, P = 16, 4
N_SIDE = SIZE // P
N_PATCHES = N_SIDE ** 2
PATCH_DIM = P * P

image_np = make_image("plus", cx=8, cy=8, size=SIZE)
image_t = torch.tensor(image_np).unsqueeze(0).unsqueeze(0)
patches_t = patchify(image_t, P)  # (1, 16, 16)
patches_np = patches_t[0].numpy()  # (16, 16)


def patch_rc(k):
    return divmod(k, N_SIDE)  # (row, col) in patch-grid units


def draw_image_grid(ax, k):
    """Full 16x16 image with 4x4 patch boundaries; patch k highlighted (if
    k < N_PATCHES) and all earlier patches shaded as already-visited."""
    ax.imshow(image_np, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, SIZE, P), minor=True)
    ax.set_yticks(np.arange(-0.5, SIZE, P), minor=True)
    ax.grid(which="minor", color="#888888", linewidth=1.0)
    ax.set_xticks([]); ax.set_yticks([])
    for j in range(min(k, N_PATCHES)):
        r, c = patch_rc(j)
        ax.add_patch(Rectangle((c * P - 0.5, r * P - 0.5), P, P, fill=False,
                                 edgecolor=DONE_COLOR, linewidth=1.4, zorder=3))
    if k < N_PATCHES:
        r, c = patch_rc(k)
        ax.add_patch(Rectangle((c * P - 0.5, r * P - 0.5), P, P, facecolor=ACTIVE_COLOR,
                                 alpha=0.28, edgecolor=ACTIVE_COLOR, linewidth=2.6, zorder=4))
    ax.set_title("16x16 image, 4x4 patches", fontsize=11)


def draw_patch_zoom(ax, k):
    ax.axis("off")
    if k >= N_PATCHES:
        return
    patch_2d = patches_np[k].reshape(P, P)
    axp = ax.inset_axes([0.15, 0.3, 0.7, 0.6])
    axp.imshow(patch_2d, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    axp.set_xticks(np.arange(-0.5, P, 1), minor=True)
    axp.set_yticks(np.arange(-0.5, P, 1), minor=True)
    axp.grid(which="minor", color=ACTIVE_COLOR, linewidth=1.2)
    axp.set_xticks([]); axp.set_yticks([])
    for spine in axp.spines.values():
        spine.set_edgecolor(ACTIVE_COLOR); spine.set_linewidth(2.2)
    ax.set_title(f"patch {k + 1}/{N_PATCHES} -> flatten to {PATCH_DIM}x1", fontsize=10,
                 color=ACTIVE_COLOR, pad=10)


def draw_sequence_matrix(ax, k):
    # patches as COLUMNS, pixels as rows -- matches vit_pipeline_demo.gif's P
    # matrix (and the W @ P convention used there), so the two animations
    # agree on what a "patch sequence" looks like instead of transposing it.
    display = np.full((PATCH_DIM, N_PATCHES), np.nan)
    filled = min(k, N_PATCHES)
    display[:, :filled] = patches_np[:filled].T
    masked = np.ma.masked_invalid(display)
    cmap = plt.get_cmap("gray_r").copy()
    cmap.set_bad(PENDING_FACE)
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_yticks([]); ax.set_xticks(range(N_PATCHES))
    ax.set_xticklabels([f"{i + 1}" for i in range(N_PATCHES)], fontsize=6.5, rotation=90)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color("#999999")
    if filled > 0 and filled <= N_PATCHES:
        last = filled - 1
        ax.add_patch(Rectangle((last - 0.5, -0.5), 1, PATCH_DIM, fill=False,
                                 edgecolor=MATCH_COLOR, linewidth=2.4, zorder=5))
    ax.set_title(f"patch sequence\n({PATCH_DIM}, {N_PATCHES})", fontsize=10.5)


def render(k, done=False):
    """k: 0..N_PATCHES-1 while stepping through patches. done=True holds the
    final frame (all 16 patches filled in) with a closing caption instead of
    the in-progress "step x/16" caption."""
    fig = plt.figure(figsize=(11.2, 5.4), dpi=120)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.75, 0.75],
                           left=0.05, right=0.97, top=0.68, bottom=0.08, wspace=0.35)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_seq = fig.add_subplot(gs[0, 2])
    draw_image_grid(ax_img, k)
    draw_patch_zoom(ax_zoom, k)
    draw_sequence_matrix(ax_seq, k + 1)
    if done:
        fig.suptitle("all 16 patches done -- one column each in the patch sequence",
                     fontsize=13, fontweight="bold", color=MATCH_COLOR, y=0.93)
        #fig.text(0.5, 0.84, "next: each column gets linearly embedded into a new vector --",
        #          ha="center", fontsize=9.5, color="#666666")
        #fig.text(0.5, 0.79, "see the pipeline animation below for what happens after that",
        #          ha="center", fontsize=9.5, color="#666666")
    else:
        fig.suptitle(f"patch {k + 1}/{N_PATCHES}: patchify, raster order",
                     fontsize=13, fontweight="bold", color=NEUTRAL_COLOR, y=0.93)
        fig.text(0.5, 0.84, "each 4x4 patch is flattened to one column of the patch sequence",
                  ha="center", fontsize=9.5, color="#666666")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 28
HOLD_END = 90
for k in range(N_PATCHES):
    frames += [render(k)] * HOLD_STEP
frames += [render(N_PATCHES - 1, done=True)] * HOLD_END

frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=55, loop=0)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
