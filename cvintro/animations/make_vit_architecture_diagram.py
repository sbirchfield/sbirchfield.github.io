"""Generate vit_architecture_diagram.png: a static (non-animated) companion
to the lesson 47 notebook's "The [CLS] token, and assembling a full ViT"
section. A single box-and-arrow flowchart of ToyViT's entire forward pass,
top to bottom, with every tensor's shape labeled at each stage -- meant to
be looked at all at once, as a quick-reference complement to
vit_pipeline_demo.gif's step-by-step trace of the same pipeline.

Deliberately distinguishes, by color, the raw post-patchify sequence
(T=16, D=16 -- NOT YET embedded, NOT Lesson 46's X) from the sequence that
actually IS Lesson 46's X (T=17, D=32, after linear embedding + prepending
[CLS] + positional encoding) -- the same distinction made explicit in the
pipeline animation and in the lesson's own prose.

Shapes match the notebook's own ToyViT exactly (cell id d269aebd:
img_size=16, patch_size=4, embed_dim=32, n_heads=4, n_layers=2). This is a
structural diagram, not a numeric one -- no model weights or data are
involved, unlike the other animations in this lesson which show real,
computed values.

Run from anywhere:
    python make_vit_architecture_diagram.py
Output:
    vit_architecture_diagram.png  (written next to this script)
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(OUT_DIR, "vit_architecture_diagram.png")

NEUTRAL_COLOR = "#333333"
ARROW_COLOR = "#555555"
OP_COLOR = "#666666"

BOX_FACE = "#eef3fa"
BOX_EDGE = "#3a6fb0"
NOTX_FACE = "#fdebd3"
NOTX_EDGE = "#e07b00"
X_FACE = "#eafaf1"
X_EDGE = "#2e9e44"
CLS_COLOR = "#c0392b"

# (label_lines, shape_text, face, edge, text_color) for each data box, top to bottom
BOXES = [
    (["input image"], "(1, 1, 16, 16)", BOX_FACE, BOX_EDGE, NEUTRAL_COLOR),
    (["raw patch sequence"], "(16, 16)  -- NOT embedded, NOT X", NOTX_FACE, NOTX_EDGE, NOTX_EDGE),
    (["linearly embedded patches"], "(32, 16)", BOX_FACE, BOX_EDGE, NEUTRAL_COLOR),
    (["+ [CLS] token, + positional encoding", "(needed only because this is classification --",
      "other tasks drop [CLS] and read the patch tokens instead)"],
     "(32, 17)  =  Lesson 46's X", X_FACE, X_EDGE, X_EDGE),
    (["Transformer encoder", "(2 layers: self-attn, add&norm,", "feed-forward, add&norm -- each",
      "shape-preserving)"], "(32, 17)  ->  (32, 17)", BOX_FACE, BOX_EDGE, NEUTRAL_COLOR),
    (["[CLS] row extracted"], "(32,)", BOX_FACE, BOX_EDGE, NEUTRAL_COLOR),
    (["classification head: nn.Linear(32, 2)"], "(2,)  -- one logit per class, softmax-normalized",
     BOX_FACE, BOX_EDGE, NEUTRAL_COLOR),
]

# operation labels drawn on the arrow between box i and box i+1
OPS = [
    "patchify: 4x4 non-overlapping patches, flattened",
    "x = W @ P: ONE shared (32,16) matrix W times every patch-column of P",
    "prepend learned cls_token; add pos_enc",
    "run the sequence through the encoder",
    "read off row 0 -- the [CLS] token's final state",
    "nn.Linear(32, 2)",
]

fig, ax = plt.subplots(figsize=(7.2, 9.6), dpi=150)
ax.set_xlim(0, 1)
ax.axis("off")

LINE_H = 0.019
PAD = 0.016
GAP = 0.075
TOP_MARGIN = 0.10
BOTTOM_MARGIN = 0.05  # room for the last box's shape-label line below it

box_heights = [2 * PAD + max(1, len(lines)) * LINE_H for lines, *_ in BOXES]
n = len(BOXES)
total_h = sum(box_heights) + (n - 1) * GAP
y_top = 1 - TOP_MARGIN
# size ylim to the actual content height instead of a fixed (0, 1) -- the
# fixed range used to be shorter than total_h, silently clipping the bottom
# box and its shape-label text against the axes' bounding box (bbox_inches
# "tight" only grows to fit artists *inside* the axes clip, not past it).
ax.set_ylim(y_top - total_h - BOTTOM_MARGIN, 1)

centers = []
box_shapes = []
box_colors = []
y = y_top
for i, (lines, shape_text, face, edge, text_color) in enumerate(BOXES):
    h = box_heights[i]
    y_box = y - h
    box = FancyBboxPatch((0.08, y_box), 0.84, h, boxstyle="round,pad=0.012",
                           facecolor=face, edgecolor=edge, linewidth=2.0, zorder=3)
    ax.add_patch(box)
    top_line_y = y - PAD - LINE_H * 0.55
    if len(lines) == 1:
        ax.text(0.5, y_box + h / 2, lines[0], ha="center", va="center", fontsize=10.5,
                 color=text_color, fontweight="bold", zorder=4)
    else:
        ax.text(0.5, top_line_y, lines[0], ha="center", va="center", fontsize=9.5,
                 color=text_color, fontweight="bold", zorder=4)
        for j, extra in enumerate(lines[1:]):
            ax.text(0.5, top_line_y - (j + 1) * LINE_H, extra, ha="center", va="center",
                     fontsize=8, color=text_color, zorder=4)
    box_shapes.append(shape_text)
    box_colors.append(text_color)
    centers.append((y, y_box))
    y = y_box - GAP

for i, op_text in enumerate(OPS):
    y_above_box = centers[i][1]
    y_below_box_top = centers[i + 1][0]
    y_mid = (y_above_box + y_below_box_top) / 2
    ax.annotate("", xy=(0.5, y_below_box_top + 0.010), xytext=(0.5, y_above_box - 0.010),
                 arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, linewidth=1.8), zorder=2)
    ax.text(0.08, y_mid, box_shapes[i], ha="left", va="center", fontsize=9,
             color=box_colors[i], style="italic", zorder=4)
    ax.text(0.56, y_mid, op_text, ha="left", va="center", fontsize=8.3, color=OP_COLOR,
             style="italic", zorder=4)

last_y_box = centers[-1][1]
ax.text(0.08, last_y_box - 0.03, box_shapes[-1], ha="left", va="center", fontsize=9,
         color=box_colors[-1], style="italic", zorder=4)

ax.text(0.5, 1 - 0.018, "ToyViT: the whole forward pass, one picture", ha="center", va="top",
         fontsize=13.5, fontweight="bold", color=NEUTRAL_COLOR)
ax.text(0.5, 1 - 0.045, "img_size=16, patch_size=4, embed_dim=32, n_heads=4, n_layers=2",
         ha="center", va="top", fontsize=9, color="#777777")

fig.savefig(OUT_PNG, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT_PNG}")
