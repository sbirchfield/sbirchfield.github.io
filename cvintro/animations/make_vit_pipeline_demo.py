"""Generate vit_pipeline_demo.gif: an animated companion to the lesson 47
notebook's "The [CLS] token, and assembling a full ViT" section.

Design rule (this is the second rewrite of this animation): ONE persistent
horizontal row of token slots stays on screen, in the same place, for the
entire pipeline from "patches just landed" through the final classification-
head read-out. Every stage acts ON that row -- the row's content changes (raw
pixels -> embedded values -> +CLS -> +positional encoding -> encoder layer
1's output -> encoder layer 2's output -> read off by the head) but its
position never jumps, and nothing is ever drawn by wiping the canvas and
starting over. A separate "explanation zone" above the row holds whatever is
explaining the CURRENT transformation (the source image, a real W @ P matrix
multiply, a thin positional-encoding strip, attention arcs) and always fades
out before the next thing fades in, so the row is the one thing the eye never
loses. A thin gray caption just below the row always states the row's current
(T, D) shape plus a one-line semantic note, so the shape story is legible at
a glance in every stage, not just in the titles.

This is an INFERENCE-ONLY animation -- it traces one forward pass and stops
at the classification head's two real logits and their softmax
probabilities. It does not train anything or compute a loss (that's covered
directly in the lesson's own prose instead, cell eb976a68).

Stages:
  1. the raw 16x16 image, then its 4x4 patch grid
  2. the 16 patches physically fly from their 2D grid positions down into
     the row -- real pixel content the whole time -- landing on the raw
     (16, 16) patch sequence, explicitly marked as NOT YET embedded
  3. the explanation zone dissolves from the (now dimmed) image into a real
     matrix multiply: the actual (32, 16) weight matrix W times the patches
     stacked as columns of a (16, 16) matrix P -- same W for every column,
     i.e. shared weights, the same idea as a reused conv kernel. P isn't
     just asserted: ghost (semi-transparent) copies of the row's own patch
     thumbnails physically fly UP from their row slots into their matching
     column inside P, so "row patch k == P's column k" is shown through
     motion, not just claimed in a label. As W @ P is revealed above, the
     row below crossfades in place from raw pixel thumbnails to real
     embedded color swatches. Each matrix box carries a small gray gloss
     line clarifying what its rows/columns actually mean (feature channels,
     patch count, etc.), not just their bare shape.
  4. a learned [CLS] swatch (its own real values, not computed from the
     image) flies in from off-canvas and prepends itself to the row
  5. positional encoding is added as a literal partial sum, column by
     column, in real time -- the row's values become exactly
     seq_with_cls + t * pos_enc for the animated t, not a cosmetic
     crossfade, so every intermediate frame is itself a real tensor, ending
     on the real (17, 32) X from Lesson 46
  6. each encoder layer: real computed attention weights (layer
     .self_attn(..., need_weights=True)) drawn as curved arcs ABOVE the row
     connecting slot positions (line opacity/width = real attention weight;
     [CLS]'s own attention row highlighted in red) -- the arcs fade out,
     then the row hard-cuts to that layer's real output (no fabricated
     in-between values, since a nonlinear layer has no meaningful linear
     path between its input and output)
  7. the [CLS] column detaches from the row and rises into the
     classification head, producing two real logits (one per class: plus,
     circle) and their softmax-normalized probabilities -- the animation
     ends here, on real inference output.

Reuses the notebook's own ToyViT verbatim (cell id d269aebd: img_size=16,
patch_size=4, in_ch=1, embed_dim=32, n_heads=4, n_layers=2) and its own
make_image('plus', ...) shape (cell id a0f686af) for the example image.
Weights are freshly initialized (seed=0), not trained.

Run from anywhere:
    python make_vit_pipeline_demo.py
Output:
    vit_pipeline_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "vit_pipeline_demo.gif")

ACTIVE_COLOR = "#e07b00"
CLS_COLOR = "#c0392b"
NEUTRAL_COLOR = "#333333"
DIM_COLOR = "#999999"
BOX_FACE = "#eef3fa"
BOX_EDGE = "#3a6fb0"
NET_COLOR = "#2e6da4"


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


def positional_encoding(T, D):
    pos = torch.arange(T).unsqueeze(1).float()
    i = torch.arange(D).unsqueeze(0).float()
    angle_rates = 1.0 / (10000 ** (2 * (i // 2) / D))
    angles = pos * angle_rates
    pe = torch.zeros(T, D)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe


class ToyViT(nn.Module):
    def __init__(self, img_size=16, patch_size=4, in_ch=1, embed_dim=32, n_heads=4, n_layers=2):
        super().__init__()
        self.patch_size = patch_size
        n_patches = (img_size // patch_size) ** 2
        patch_dim = in_ch * patch_size * patch_size
        self.embed = nn.Linear(patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.register_buffer("pos_enc", positional_encoding(n_patches + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(embed_dim, n_heads, dim_feedforward=embed_dim * 2,
                                            batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, x):
        p = patchify(x, self.patch_size)
        tok = self.embed(p)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tok = torch.cat([cls, tok], dim=1) + self.pos_enc
        out = self.encoder(tok)
        return self.head(out[:, 0])


SIZE, P, EMBED_DIM, N_LAYERS = 16, 4, 32, 2
N_SIDE = SIZE // P
N_PATCHES = N_SIDE ** 2

torch.manual_seed(0)
model = ToyViT(img_size=SIZE, patch_size=P, embed_dim=EMBED_DIM, n_layers=N_LAYERS)
model.eval()

image_np = make_image("plus", cx=8, cy=8, size=SIZE)
image_t = torch.tensor(image_np).unsqueeze(0).unsqueeze(0)

layer0 = model.encoder.layers[0]
layer1 = model.encoder.layers[1]

with torch.no_grad():
    patches_t = patchify(image_t, P)                       # (1, 16, 16)
    W_embed = model.embed.weight.numpy()                    # (32, 16) -- SAME matrix for every patch
    b_embed = model.embed.bias.numpy()                      # (32,)
    embedded = model.embed(patches_t)                       # (1, 16, 32)
    cls = model.cls_token.expand(1, -1, -1)                 # (1, 1, 32)
    seq_with_cls = torch.cat([cls, embedded], dim=1)        # (1, 17, 32)
    seq_pos = seq_with_cls + model.pos_enc                  # (1, 17, 32)

    attn_out1, attn_w1 = layer0.self_attn(seq_pos, seq_pos, seq_pos,
                                           need_weights=True, average_attn_weights=True)
    layer1_out = layer0(seq_pos)
    attn_out2, attn_w2 = layer1.self_attn(layer1_out, layer1_out, layer1_out,
                                           need_weights=True, average_attn_weights=True)
    layer2_out = layer1(layer1_out)
    encoded = model.encoder(seq_pos)
    assert torch.allclose(layer2_out, encoded, atol=1e-5), "layer trace != model.encoder(seq_pos)"

    cls_out = encoded[:, 0]
    logits = model.head(cls_out)             # (1, 2) -- one logit per class
    probs = F.softmax(logits, dim=-1)         # (1, 2)

patches_np = patches_t[0].numpy()              # (16, 16)
embedded_np = embedded[0].numpy()               # (16, 32)
cls_vec_np = cls[0, 0].detach().numpy()         # (32,) -- the raw learned CLS vector, pre-pos-enc
seq_with_cls_np = seq_with_cls[0].numpy()       # (17, 32)
pos_enc_np = model.pos_enc.numpy()              # (17, 32)
seq_pos_np = seq_pos[0].numpy()                 # (17, 32)
attn_w1_np = attn_w1[0].numpy()                 # (17, 17)
layer1_out_np = layer1_out[0].numpy()           # (17, 32)
attn_w2_np = attn_w2[0].numpy()                 # (17, 17)
layer2_out_np = layer2_out[0].numpy()           # (17, 32)
cls_out_np = cls_out[0].numpy()                 # (32,)
logits_np = logits[0].numpy()                   # (2,) -- one logit per class
probs_np = probs[0].numpy()                     # (2,) -- softmax-normalized
P_cols = patches_np.T                           # (16, 16) -- each column is one patch's 16-vector
pe_strip_np = pos_enc_np.T                      # (32, 17) -- for the thin pos-enc strip visual

VMAX = max(1.0, max(float(np.abs(arr).max()) for arr in
                     [embedded_np, seq_with_cls_np, seq_pos_np, layer1_out_np, layer2_out_np]))

print(f"logits={logits_np}, probs={probs_np}")

# --- layout: one persistent row, one explanation zone above it ---------
XLIM, YLIM = 13.0, 8.0
SLOT_W, SLOT_SPACING = 0.62, 0.72
ROW_Y0, ROW_Y1 = 1.0, 2.1
ROW_YC = (ROW_Y0 + ROW_Y1) / 2
ROW_H = ROW_Y1 - ROW_Y0
IMG_SIZE = 4.0
IMG_X0 = XLIM / 2 - IMG_SIZE / 2
IMG_Y0 = 3.1

# the P matrix box's geometry in stage 4 -- shared between the static box
# draw and the ghost-fly destination computation, so they always agree
P_BOX_CX, P_BOX_CY, P_BOX_W, P_BOX_H = 6.4, 5.0, 2.2, 2.2


def slot_x(k, n):
    start = (XLIM - n * SLOT_SPACING) / 2 + SLOT_W / 2
    return start + k * SLOT_SPACING


def grid_center(k):
    r, c = divmod(k, N_SIDE)
    cx = IMG_X0 + c + 0.5
    cy = IMG_Y0 + IMG_SIZE - r - 0.5
    return cx, cy


def p_col_pos(k):
    """Center of column k inside the P matrix box (stage 4's ghost-fly target)."""
    left = P_BOX_CX - P_BOX_W / 2
    col_w = P_BOX_W / N_PATCHES
    return left + (k + 0.5) * col_w, P_BOX_CY


def ease(t):
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def clear_ax(ax):
    ax.set_xlim(0, XLIM)
    ax.set_ylim(0, YLIM)
    ax.axis("off")


def draw_image(ax, with_grid=False, dim=False, alpha=1.0):
    base_alpha = (0.35 if dim else 1.0) * alpha
    if base_alpha <= 0.01:
        return
    extent = (IMG_X0, IMG_X0 + IMG_SIZE, IMG_Y0, IMG_Y0 + IMG_SIZE)
    ax.imshow(image_np, cmap="gray_r", vmin=0, vmax=1, extent=extent, zorder=2, alpha=base_alpha,
              aspect="auto")
    ax.add_patch(Rectangle((IMG_X0, IMG_Y0), IMG_SIZE, IMG_SIZE, fill=False,
                            edgecolor="#999999", linewidth=1.2, zorder=3, alpha=min(1.0, alpha)))
    if with_grid:
        for i in range(1, N_SIDE):
            ax.plot([IMG_X0 + i, IMG_X0 + i], [IMG_Y0, IMG_Y0 + IMG_SIZE],
                     color=ACTIVE_COLOR, linewidth=1.0, zorder=3, alpha=alpha)
            ax.plot([IMG_X0, IMG_X0 + IMG_SIZE], [IMG_Y0 + i, IMG_Y0 + i],
                     color=ACTIVE_COLOR, linewidth=1.0, zorder=3, alpha=alpha)
        for k in range(N_PATCHES):
            cx, cy = grid_center(k)
            ax.text(cx, cy, str(k + 1), ha="center", va="center", fontsize=7.5,
                     color=ACTIVE_COLOR, fontweight="bold", zorder=4, alpha=alpha)


def draw_patch_thumb(ax, k, cx, cy, w, h, edgecolor=ACTIVE_COLOR, lw=1.0, alpha=1.0, zorder=5):
    if alpha <= 0.01:
        return
    extent = (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
    ax.imshow(patches_np[k].reshape(P, P), cmap="gray_r", vmin=0, vmax=1,
              extent=extent, zorder=zorder, alpha=alpha, aspect="auto")
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False,
                            edgecolor=edgecolor, linewidth=lw, zorder=zorder + 1, alpha=alpha))


def draw_col_swatch(ax, data_vec, cx, cy, w, h, vmax, alpha=1.0, highlight=False, zorder=5):
    """data_vec: (D,) -- one token's real embedding, color = sign & magnitude."""
    if alpha <= 0.01:
        return
    extent = (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
    ax.imshow(data_vec.reshape(-1, 1), cmap="RdBu_r", vmin=-vmax, vmax=vmax,
              extent=extent, aspect="auto", zorder=zorder, alpha=alpha)
    edge = CLS_COLOR if highlight else "#888888"
    lw = 2.4 if highlight else 1.0
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False,
                            edgecolor=edge, linewidth=lw, zorder=zorder + 1, alpha=alpha))


def draw_swatch_row(ax, data, n, x_positions=None, y=ROW_YC, h=ROW_H, highlight_col=None,
                     dim_cols=None, alpha=1.0, vmax=None):
    """data: (D, n) -- one real column per token."""
    vmax = vmax or VMAX
    for k in range(n):
        cx = x_positions[k] if x_positions is not None else slot_x(k, n)
        a = alpha * (0.35 if dim_cols and k in dim_cols else 1.0)
        draw_col_swatch(ax, data[:, k], cx, y, SLOT_W, h, vmax, alpha=a,
                         highlight=(highlight_col == k))


def label_row_dims(ax, n, d, note="", alpha=1.0):
    """A small, always-present gray caption just below the row stating its
    current (T, D) shape plus a short semantic note -- so the shape story is
    legible at a glance, not just inferable from the title text."""
    if alpha <= 0.01:
        return
    text = f"T={n} tokens  x  D={d} features/token"
    if note:
        text += f"   --   {note}"
    ax.text(XLIM / 2, ROW_Y0 - 0.32, text, ha="center", va="top", fontsize=8,
             color=DIM_COLOR, zorder=6, alpha=alpha)


def draw_substeps(ax, steps, active_idx, x=0.4, y_top=7.6, line_h=0.32, fontsize=9.5):
    """A persistent, downward-accumulating checklist of a stage's sub-steps --
    earlier entries stay on screen (dimmed), the current one is highlighted,
    later ones are faint-but-visible, so the whole sequence is legible from
    the first frame instead of being overwritten frame to frame."""
    for i, step in enumerate(steps):
        y = y_top - i * line_h
        if i < active_idx:
            color, alpha, prefix = DIM_COLOR, 0.55, "✓ "
        elif i == active_idx:
            color, alpha, prefix = NEUTRAL_COLOR, 1.0, "▸ "
        else:
            color, alpha, prefix = DIM_COLOR, 0.3, "  "
        weight = "bold" if i == active_idx else "normal"
        ax.text(x, y, prefix + step, ha="left", va="top", fontsize=fontsize, color=color,
                alpha=alpha, fontweight=weight, zorder=9)


def draw_matrix_box(ax, data, cx, cy, w, h, vmax, label, cmap="RdBu_r", vmin=None, alpha=1.0,
                     gloss=None):
    if alpha <= 0.01:
        return
    extent = (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
    ax.imshow(data, cmap=cmap, vmin=(vmin if vmin is not None else -vmax), vmax=vmax,
              extent=extent, aspect="auto", zorder=4, alpha=alpha)
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False,
                            edgecolor=BOX_EDGE, linewidth=1.6, zorder=5, alpha=alpha))
    label_y = cy - h / 2 - 0.22
    ax.text(cx, label_y, label, ha="center", va="top", fontsize=8.5,
            color=NEUTRAL_COLOR, zorder=5, alpha=alpha)
    if gloss:
        n_lines = label.count("\n") + 1
        gloss_y = label_y - n_lines * 0.26 - 0.05
        ax.text(cx, gloss_y, gloss, ha="center", va="top", fontsize=7.0,
                color=DIM_COLOR, zorder=5, alpha=alpha)


def draw_pos_strip(ax, alpha, n=N_PATCHES + 1):
    if alpha <= 0.01:
        return
    y0, y1 = ROW_Y1 + 0.1, ROW_Y1 + 0.5
    for k in range(n):
        cx = slot_x(k, n)
        extent = (cx - SLOT_W / 2, cx + SLOT_W / 2, y0, y1)
        ax.imshow(pe_strip_np[:, k].reshape(-1, 1), cmap="RdBu_r", vmin=-1, vmax=1,
                  extent=extent, aspect="auto", zorder=5, alpha=alpha)
    ax.text(XLIM / 2, y1 + 0.28, "+", ha="center", fontsize=16, color=NEUTRAL_COLOR,
             alpha=alpha, zorder=6)


def quad_bezier(p0, p1, p2, steps=24):
    t = np.linspace(0, 1, steps)
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
    return x, y


def draw_attn_arcs(ax, attn_w, n, alpha=1.0):
    if alpha <= 0.01:
        return
    amax = max(float(attn_w.max()), 1e-6)
    for i in range(n):
        for j in range(i + 1, n):
            val = (attn_w[i, j] + attn_w[j, i]) / 2
            xi, xj = slot_x(i, n), slot_x(j, n)
            h = min(3.6, 0.25 + 0.22 * (j - i))
            mx = (xi + xj) / 2
            x, y = quad_bezier((xi, ROW_Y1), (mx, ROW_Y1 + h), (xj, ROW_Y1))
            ax.plot(x, y, color=NET_COLOR, alpha=min(0.35, 0.5 * val / amax) * alpha,
                     linewidth=0.8, zorder=2)
    cmax = max(float(attn_w[0].max()), 1e-6)
    for j in range(1, n):
        val = attn_w[0, j]
        xi, xj = slot_x(0, n), slot_x(j, n)
        h = min(4.2, 0.3 + 0.24 * j)
        mx = (xi + xj) / 2
        x, y = quad_bezier((xi, ROW_Y1), (mx, ROW_Y1 + h), (xj, ROW_Y1))
        ax.plot(x, y, color=CLS_COLOR, alpha=min(0.9, 0.25 + 0.65 * val / cmax) * alpha,
                 linewidth=(1.0 + 2.0 * val / cmax), zorder=3)


def render(draw_fn, title, subtitle=""):
    fig, ax = plt.subplots(figsize=(12.0, 7.2), dpi=118)
    clear_ax(ax)
    draw_fn(ax)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=NEUTRAL_COLOR, y=0.975)
    if subtitle:
        fig.text(0.5, 0.905, subtitle, ha="center", fontsize=9.5, color="#666666")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.02)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD = 58
HOLD_SHORT = 40


def add(draw_fn, title, subtitle="", hold=HOLD):
    frames.extend([render(draw_fn, title, subtitle)] * hold)


# 1. plain image
add(lambda ax: draw_image(ax, with_grid=False), "stage 1/8: the input image",
    "a 16x16 binary image of a plus shape")

# 2. patch grid
add(lambda ax: draw_image(ax, with_grid=True), "stage 2/8: patchify",
    "Non-overlapping 4x4 patches, numbered in raster order")

# 3. fly: patches travel from grid down into the row
STAGE3_TITLE = "stage 3/8: the raw patch sequence"
STAGE3_SUBTITLE = "serialize the patches in raster order"

N_FLY = 14
for f in range(N_FLY + 1):
    t = ease(f / N_FLY)

    def draw_fly(ax, t=t):
        draw_image(ax, with_grid=True, dim=True)
        for k in range(N_PATCHES):
            gx, gy = grid_center(k)
            sx, sy = slot_x(k, N_PATCHES), ROW_YC
            cx, cy = lerp(gx, sx, t), lerp(gy, sy, t)
            w, h = lerp(1.0, SLOT_W, t), lerp(1.0, ROW_H, t)
            draw_patch_thumb(ax, k, cx, cy, w, h)
    frames.append(render(draw_fly, STAGE3_TITLE, STAGE3_SUBTITLE))


def draw_raw_row(ax, img_alpha=1.0):
    draw_image(ax, with_grid=True, dim=True, alpha=img_alpha)
    for k in range(N_PATCHES):
        draw_patch_thumb(ax, k, slot_x(k, N_PATCHES), ROW_YC, SLOT_W, ROW_H)
    label_row_dims(ax, N_PATCHES, P * P, "raw pixel values")


add(draw_raw_row, STAGE3_TITLE, STAGE3_SUBTITLE)
frames.extend([frames[-1]] * 20)

# 4. the explanation zone dissolves image -> matrix multiply; ghost copies of
# the row's own patches fly up into P's matching columns, proving the
# correspondence through motion; the row crossfades to embeddings in place.
EMBED_OUT = W_embed @ P_cols + b_embed[:, None]

def draw_embed_transition(ax, img_fade, mat_fade, reveal_t, ghost_t=1.0, phase_idx=0):
    draw_image(ax, with_grid=True, dim=True, alpha=img_fade)
    if mat_fade > 0.01:
        draw_matrix_box(ax, W_embed, 3.0, 5.5, 2.4, 4.0, float(np.abs(W_embed).max()),
                         "W  (32, 16)\nsingle weight matrix", alpha=mat_fade,
                         gloss="32 rows=embed_dim (out)   16 cols=patch_dim (in)")
        ax.text(5.0, 5.0, "x", ha="center", va="center", fontsize=20, color=NEUTRAL_COLOR,
                 zorder=5, alpha=mat_fade)
        # The P box's border/label are drawn once and never swapped out. The
        # crisp final (16,16) matrix crossfades in underneath the flying
        # thumbnails over the last part of the flight, instead of popping in
        # abruptly the instant ghost_t crosses a threshold -- that instant
        # swap (blurry stretched thumbnails -> a completely different crisp
        # grid, held for many frames) was the actual jarring cut, not a
        # motion-speed problem.
        ax.add_patch(Rectangle((P_BOX_CX - P_BOX_W / 2, P_BOX_CY - P_BOX_H / 2),
                                P_BOX_W, P_BOX_H, fill=False, edgecolor=BOX_EDGE,
                                linewidth=1.2, zorder=4, alpha=mat_fade))
        ax.text(P_BOX_CX, P_BOX_CY - P_BOX_H / 2 - 0.22, "P  (16, 16)\npatches, one per column",
                 ha="center", va="top", fontsize=8.5, color=NEUTRAL_COLOR, zorder=5,
                 alpha=mat_fade)
        ax.text(P_BOX_CX, P_BOX_CY - P_BOX_H / 2 - 0.74,
                 "16 cols=T=patches   16 rows=pixels/patch", ha="center", va="top",
                 fontsize=7.0, color=DIM_COLOR, zorder=5, alpha=mat_fade)

        gt = ease(min(1.0, ghost_t))
        settle = max(0.0, min(1.0, (gt - 0.65) / 0.35))
        if settle > 0.01:
            extent = (P_BOX_CX - P_BOX_W / 2, P_BOX_CX + P_BOX_W / 2,
                      P_BOX_CY - P_BOX_H / 2, P_BOX_CY + P_BOX_H / 2)
            ax.imshow(P_cols, cmap="gray_r", vmin=0, vmax=1, extent=extent, aspect="auto",
                      zorder=4, alpha=mat_fade * settle)
        fly_alpha = 0.55 * mat_fade * (1.0 - settle)
        if fly_alpha > 0.01:
            for k in range(N_PATCHES):
                gx0, gy0 = slot_x(k, N_PATCHES), ROW_YC
                gx1, gy1 = p_col_pos(k)
                gx, gy = lerp(gx0, gx1, gt), lerp(gy0, gy1, gt)
                # collapses from a square thumbnail into a narrow vertical
                # vector in flight, landing already shaped like the column
                # it's about to become -- the crossfade onto the crisp real
                # P matrix (above) hides the seam, so this no longer pops.
                gw = lerp(SLOT_W, P_BOX_W / N_PATCHES, gt)
                gh = lerp(ROW_H, P_BOX_H, gt)
                draw_patch_thumb(ax, k, gx, gy, gw, gh, edgecolor=BOX_EDGE, lw=0.5,
                                  alpha=fly_alpha, zorder=6)
        ax.text(8.1, 5.0, "=", ha="center", va="center", fontsize=20, color=NEUTRAL_COLOR,
                 zorder=5, alpha=mat_fade)
        out_alpha = mat_fade * (0.12 + 0.88 * reveal_t)
        draw_matrix_box(ax, EMBED_OUT, 10.4, 5.0, 2.2, 4.0, VMAX,
                         ("W x P  (32, 16)" if reveal_t > 0.4 else "..."), alpha=out_alpha,
                         gloss="32 rows=embed_dim   16 cols=T=tokens (embedded)")
    thumb_alpha = max(0.0, 1.0 - reveal_t)
    if thumb_alpha > 0.01:
        for k in range(N_PATCHES):
            draw_patch_thumb(ax, k, slot_x(k, N_PATCHES), ROW_YC, SLOT_W, ROW_H, alpha=thumb_alpha)
    if reveal_t > 0.01:
        draw_swatch_row(ax, embedded_np.T, N_PATCHES, alpha=reveal_t)
    # Caption text is driven by the discrete phase, not by reveal_t crossing a
    # threshold mid-crossfade -- a threshold-triggered flip mid-loop caused a
    # jarring instant re-center of this (center-anchored) text; now it only
    # ever changes once per phase, at a phase BOUNDARY, never mid-motion.
    if phase_idx == 0:
        label_row_dims(ax, N_PATCHES, P * P, "raw pixel values")
    else:
        label_row_dims(ax, N_PATCHES, EMBED_DIM, "after embedding")


STAGE4_TITLE = "stage 4/8: linear embedding"
STAGE4_SUBTITLE = "same weights W applied identically to every patch via a single matrix multiply:  W x P"

IMG_FADE_FRAC = 0.35  # the image clears out of the P box's screen region FIRST, completely,
# before the matrix diagram/ghost-fly start fading in -- they used to crossfade
# simultaneously in the same screen space (the image sits centered right where
# P's box and the ghost-fly destination are), which looked like two unrelated
# pictures colliding. Staggering them removes the overlap entirely.
N_GHOST = 22
for f in range(N_GHOST + 1):
    gt = f / N_GHOST
    img_t = min(1.0, gt / IMG_FADE_FRAC)
    mat_t = max(0.0, (gt - IMG_FADE_FRAC) / (1 - IMG_FADE_FRAC))
    img_fade = 1.0 - ease(img_t)
    mat_fade = ease(mat_t)
    frames.append(render(lambda ax, img_fade=img_fade, mat_fade=mat_fade, mat_t=mat_t:
                          draw_embed_transition(ax, img_fade, mat_fade, 0.0, ghost_t=mat_t,
                                                 phase_idx=0),
                          STAGE4_TITLE, STAGE4_SUBTITLE))
frames.extend([frames[-1]] * 18)

N_REVEAL = 14
for f in range(N_REVEAL + 1):
    t = ease(f / N_REVEAL)
    frames.append(render(lambda ax, t=t: draw_embed_transition(ax, 0.0, 1.0, t, ghost_t=1.0,
                                                                   phase_idx=1),
                          STAGE4_TITLE, STAGE4_SUBTITLE))
frames.extend([frames[-1]] * 26)

N_FADE = 10
for f in range(N_FADE + 1):
    t = f / N_FADE
    frames.append(render(lambda ax, t=t: draw_embed_transition(ax, 0.0, 1 - t, 1.0, ghost_t=1.0,
                                                                   phase_idx=2),
                          STAGE4_TITLE, STAGE4_SUBTITLE))
frames.extend([frames[-1]] * 16)

# 5. CLS flies in -- into the EMBEDDED row, not the raw pixels
STAGE5_TITLE = "stage 5/8: prepend the [CLS] token"
STAGE5_SUBTITLE = "[CLS] is a free parameter, learned during training"

N_CLS_FLY = 12
for f in range(N_CLS_FLY + 1):
    t = ease(f / N_CLS_FLY)

    def draw_cls_fly(ax, t=t):
        xs = [lerp(slot_x(k, N_PATCHES), slot_x(k + 1, N_PATCHES + 1), t) for k in range(N_PATCHES)]
        draw_swatch_row(ax, embedded_np.T, N_PATCHES, x_positions=xs)
        cx = lerp(-1.0, slot_x(0, N_PATCHES + 1), t)
        draw_col_swatch(ax, cls_vec_np, cx, ROW_YC, SLOT_W, ROW_H, VMAX, highlight=True)
    frames.append(render(draw_cls_fly, STAGE5_TITLE, STAGE5_SUBTITLE))
frames.extend([frames[-1]] * 20)


def draw_cls_settled(ax):
    xs = [slot_x(k + 1, N_PATCHES + 1) for k in range(N_PATCHES)]
    draw_swatch_row(ax, embedded_np.T, N_PATCHES, x_positions=xs)
    draw_col_swatch(ax, cls_vec_np, slot_x(0, N_PATCHES + 1), ROW_YC, SLOT_W, ROW_H, VMAX,
                     highlight=True)
    label_row_dims(ax, N_PATCHES + 1, EMBED_DIM, "1 [CLS] token + 16 patch tokens")


add(draw_cls_settled, STAGE5_TITLE, STAGE5_SUBTITLE)

# 6. positional encoding: a literal partial sum, not a cosmetic crossfade
def draw_pos_intro(ax, a):
    draw_swatch_row(ax, seq_with_cls_np.T, N_PATCHES + 1, highlight_col=0)
    draw_pos_strip(ax, a)
    label_row_dims(ax, N_PATCHES + 1, EMBED_DIM, "about to add positional encoding")


N_POS_IN = 10
for f in range(N_POS_IN + 1):
    a = f / N_POS_IN
    frames.append(render(lambda ax, a=a: draw_pos_intro(ax, a),
                          "stage 6/8: add positional encoding",
                          "a fixed sinusoidal pattern is added to the tokens, column by column"))
frames.extend([frames[-1]] * 18)

N_POS_MERGE = 14
for f in range(N_POS_MERGE + 1):
    t = ease(f / N_POS_MERGE)
    partial = seq_with_cls_np + t * pos_enc_np  # exact partial sum -- a real tensor at every t

    def draw_pos_merge(ax, t=t, partial=partial):
        draw_swatch_row(ax, partial.T, N_PATCHES + 1, highlight_col=0)
        draw_pos_strip(ax, 1 - t)
    frames.append(render(draw_pos_merge, "stage 6/8: add positional encoding",
                          "a fixed sinusoidal pattern is added to the tokens, column by column"))
frames.extend([frames[-1]] * 24)


# 7. each encoder layer: real attention arcs above the row, then a hard cut to the real output
STAGE7_STEPS = [
    "self-attention (drawn as arcs , with [CLS]'s row in red)",
    "feed-forward pass",
]


def draw_mix(ax, attn_w, row_data, arc_alpha):
    draw_swatch_row(ax, row_data.T, N_PATCHES + 1, highlight_col=0)
    draw_attn_arcs(ax, attn_w, N_PATCHES + 1, alpha=arc_alpha)
    label_row_dims(ax, N_PATCHES + 1, EMBED_DIM, "self-attention mixing in progress")
    draw_substeps(ax, STAGE7_STEPS, 0)


def draw_layer_out(ax, row_out, label):
    draw_swatch_row(ax, row_out.T, N_PATCHES + 1, highlight_col=0)
    label_row_dims(ax, N_PATCHES + 1, EMBED_DIM, f"{label}'s output (same shape in and out)")
    draw_substeps(ax, STAGE7_STEPS, 1)


def encoder_layer_stages(label, attn_w, row_in, row_out):
    title = f"stage 7/8: {label}"
    subtitle = "self-attn -> add&norm -> feed-forward -> add&norm"
    n_in, n_out = 10, 10
    for f in range(n_in + 1):
        a = f / n_in
        frames.append(render(lambda ax, a=a: draw_mix(ax, attn_w, row_in, a), title, subtitle))
    frames.extend([frames[-1]] * 20)
    for f in range(n_out + 1):
        a = 1 - f / n_out
        frames.append(render(lambda ax, a=a: draw_mix(ax, attn_w, row_in, a), title, subtitle))
    add(lambda ax: draw_layer_out(ax, row_out, label), title, subtitle, hold=HOLD_SHORT)


encoder_layer_stages("encoder layer 1", attn_w1_np, seq_pos_np, layer1_out_np)
encoder_layer_stages("encoder layer 2", attn_w2_np, layer1_out_np, layer2_out_np)

# 8. CLS detaches and rises to the head -- two real logits and their
# softmax probabilities. Inference ends here.
# CLS lands left of center (canvas center is XLIM/2=6.5) rather than way out
# toward the right edge -- there's plenty of width, and stopping near the
# middle leaves clear room for the head box just to its right, so the arrow
# between them still reads left-to-right (the natural reading direction:
# CLS's value -> nn.Linear -> logits) without needing to crowd the edge.
CLS_STOP_X, CLS_STOP_Y = 4.6, 5.3
HEAD_CX, HEAD_CY, HEAD_W, HEAD_H = 6.9, 5.3, 1.8, 1.4
LOGIT_TX = 9.6  # logit/softmax text sits to the RIGHT of the head box, so
# CLS -> nn.Linear -> logits reads as one simple left-to-right flow, with
# no arrow ever turning downward.
N_PULL = 12
for f in range(N_PULL + 1):
    t = ease(f / N_PULL)

    def draw_pull(ax, t=t):
        draw_swatch_row(ax, layer2_out_np.T, N_PATCHES + 1, dim_cols=range(1, N_PATCHES + 1))
        cx0, cy0 = slot_x(0, N_PATCHES + 1), ROW_YC
        cx, cy = lerp(cx0, CLS_STOP_X, t), lerp(cy0, CLS_STOP_Y, t)
        w, h = lerp(SLOT_W, 0.5, t), lerp(ROW_H, 2.0, t)
        draw_col_swatch(ax, cls_out_np, cx, cy, w, h, VMAX, highlight=True)
        label_row_dims(ax, N_PATCHES + 1, EMBED_DIM, "all but [CLS] is ignored")
    frames.append(render(draw_pull, "stage 8/8: final linear layer",
                          "the classification head only sees the first [CLS] token"))
frames.extend([frames[-1]] * 20)


def draw_head_logit(ax):
    draw_swatch_row(ax, layer2_out_np.T, N_PATCHES + 1, dim_cols=range(1, N_PATCHES + 1))
    draw_col_swatch(ax, cls_out_np, CLS_STOP_X, CLS_STOP_Y, 0.5, 2.0, VMAX, highlight=True)
    ax.text(CLS_STOP_X, CLS_STOP_Y + 1.15, "(32,) -- CLS row's full embedding,\nread off from (17,32)",
             ha="center", va="bottom", fontsize=7.5, color=DIM_COLOR, zorder=8)
    ax.add_patch(Rectangle((HEAD_CX - HEAD_W / 2, HEAD_CY - HEAD_H / 2), HEAD_W, HEAD_H,
                            facecolor=BOX_FACE, edgecolor=BOX_EDGE, linewidth=1.8, zorder=7))
    ax.text(HEAD_CX, HEAD_CY, "nn.Linear\n(32, 2)", ha="center", va="center", fontsize=9, zorder=8)
    # Both arrows flow left -> right: CLS -> head box -> logits. Nothing
    # turns downward, so the whole stage reads as one simple left-to-right
    # chain instead of a bend.
    ax.annotate("", xy=(HEAD_CX - HEAD_W / 2, HEAD_CY), xytext=(CLS_STOP_X + 0.3, CLS_STOP_Y),
                arrowprops=dict(arrowstyle="-|>", color=NEUTRAL_COLOR, linewidth=1.6), zorder=7)
    ax.annotate("", xy=(LOGIT_TX - 0.3, HEAD_CY), xytext=(HEAD_CX + HEAD_W / 2, HEAD_CY),
                arrowprops=dict(arrowstyle="-|>", color=NEUTRAL_COLOR, linewidth=1.6), zorder=7)
    ax.text(LOGIT_TX, HEAD_CY + 0.3,
             f"plus logit = {logits_np[0]:.3f}\ncircle logit = {logits_np[1]:.3f}", ha="left",
             va="center", fontsize=10.5, fontweight="bold", color=ACTIVE_COLOR, zorder=8)
    ax.text(LOGIT_TX, HEAD_CY - 0.75,
             f"after softmax:\nprob(plus)={probs_np[0]:.3f}\nprob(circle)={probs_np[1]:.3f}", ha="left",
             va="center", fontsize=9, color=NEUTRAL_COLOR, zorder=8)
    label_row_dims(ax, N_PATCHES + 1, EMBED_DIM, "all but [CLS] is ignored")


add(draw_head_logit, "stage 8/8: final linear layer",
    "the classification head only sees the first [CLS] token", hold=130)

frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=55, loop=0)
print(f"Saved {OUT_GIF} ({len(frames)} frames); logits={logits_np}")

# mp4 export for fast local iteration -- scrub/step frame-by-frame in any
# video player (unlike the gif, which most viewers only auto-play). Not
# referenced by the lesson notebook; purely a dev convenience, skip with
# --no-mp4 if imageio/imageio-ffmpeg aren't installed.
if "--no-mp4" not in __import__("sys").argv:
    import imageio.v2 as imageio
    import numpy as np
    OUT_MP4 = OUT_GIF.replace(".gif", ".mp4")
    fps = round(1000 / 55)
    with imageio.get_writer(OUT_MP4, fps=fps, codec="libx264", quality=8) as writer:
        for fr in frames:
            arr = np.array(fr)
            h, w = arr.shape[:2]
            writer.append_data(arr[:h - h % 2, :w - w % 2])  # libx264 needs even dims
    print(f"Saved {OUT_MP4} ({len(frames)} frames @ {fps}fps -> {len(frames) / fps:.1f}s)")
