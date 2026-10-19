"""Generate iou_demo.gif: a conceptual companion to the lesson 41 notebook's
"The model and the IoU metric" section (IoU is introduced in lesson 40 and
reused here as the training/eval metric). Slides a "predicted" box across a
fixed "true" box, shading the intersection and union live and building up a
little IoU-vs-offset curve, so the intersection-over-union formula is seen
rather than just read.

Standalone (not tied to either lesson's specific dataset) -- uses the same
(cx, cy, w, h) box convention and intersection-over-union formula as
lesson40_object_detection_sliding_windows.ipynb's `iou()`.

Run from anywhere:
    python make_iou_demo.py
Output:
    iou_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "iou_demo.gif")

TEXT_COLOR = "#222222"
TRUE_COLOR = "#2e9e44"
PRED_COLOR = "#d62728"
INTER_COLOR = "#f2c200"
CURVE_COLOR = "#1f6fb4"

BOX_W = BOX_H = 40
TRUE_BOX = (50.0, 50.0, BOX_W, BOX_H)  # cx, cy, w, h


def box_corners(b):
    cx, cy, w, h = b
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def iou(a, b):
    ax0, ay0, ax1, ay1 = box_corners(a)
    bx0, by0, bx1, by1 = box_corners(b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    aw, ah = ax1 - ax0, ay1 - ay0
    bw, bh = bx1 - bx0, by1 - by0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0, inter, union, (ix0, iy0, ix1, iy1)


OFFSETS = list(np.arange(-55, 55.01, 2.5))
# ping-pong: sweep right across full overlap, pause, then sweep back --
# so the curve is traced out once left-to-right and the shape is clearly a
# "tent" peaking at offset 0.
history_x, history_y = [], []

# View bounds wide enough that the predicted box is always fully on-screen,
# even at the extreme offsets -- otherwise it gets clipped at the edges.
VIEW_PAD = 6
VIEW_X0 = TRUE_BOX[0] + min(OFFSETS) - BOX_W / 2 - VIEW_PAD
VIEW_X1 = TRUE_BOX[0] + max(OFFSETS) + BOX_W / 2 + VIEW_PAD
VIEW_SPAN = VIEW_X1 - VIEW_X0
VIEW_Y0 = TRUE_BOX[1] - VIEW_SPAN / 2
VIEW_Y1 = TRUE_BOX[1] + VIEW_SPAN / 2


def render(offset, freeze_hold=False):
    pred_box = (TRUE_BOX[0] + offset, TRUE_BOX[1], BOX_W, BOX_H)
    val, inter, union, ibox = iou(TRUE_BOX, pred_box)

    if not freeze_hold:
        history_x.append(offset)
        history_y.append(val)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), dpi=120,
                              gridspec_kw={"width_ratios": [1.1, 1.0]})
    fig.subplots_adjust(top=0.80, bottom=0.11, left=0.05, right=0.97, wspace=0.28)

    ax = axes[0]
    ax.set_xlim(VIEW_X0, VIEW_X1); ax.set_ylim(VIEW_Y1, VIEW_Y0)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Hatch each box's full interior -- together the two hatched rectangles
    # cover exactly the union. The intersection is then painted solid yellow
    # on top, so only the non-overlapping part of the union stays hatched.
    tx0, ty0, tx1, ty1 = box_corners(TRUE_BOX)
    ax.add_patch(patches.Rectangle((tx0, ty0), tx1 - tx0, ty1 - ty0, edgecolor=TRUE_COLOR,
                                    facecolor="none", hatch="///", linewidth=2.6, zorder=2, label="true box"))
    px0, py0, px1, py1 = box_corners(pred_box)
    # A tight custom dash pattern (rather than the default "--") keeps several
    # dashes per edge even at the small size the legend swatch renders at, so
    # the rectangle outline reads clearly there too.
    ax.add_patch(patches.Rectangle((px0, py0), px1 - px0, py1 - py0, edgecolor=PRED_COLOR,
                                    facecolor="none", hatch="///", linewidth=2.2,
                                    linestyle=(0, (3, 2)), zorder=2, label="predicted box"))

    ix0, iy0, ix1, iy1 = ibox
    if ix1 > ix0 and iy1 > iy0:
        ax.add_patch(patches.Rectangle((ix0, iy0), ix1 - ix0, iy1 - iy0,
                                        facecolor=INTER_COLOR, edgecolor="none", alpha=0.85, zorder=3))

    union_proxy = patches.Patch(facecolor="none", edgecolor="#888888", hatch="///", label="union")
    inter_proxy = patches.Patch(facecolor=INTER_COLOR, edgecolor="none", alpha=0.85, label="intersection")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [union_proxy, inter_proxy], loc="upper center",
              bbox_to_anchor=(0.5, 1.16), ncol=4, fontsize=7.8, frameon=False, handlelength=2.2,
              columnspacing=1.0)

    ax.text(0.5, -0.055, f"intersection area = {inter:5.0f}      union area = {union:5.0f}",
            transform=ax.transAxes, ha="center", fontsize=9, color=TEXT_COLOR)

    ax2 = axes[1]
    ax2.plot(history_x, history_y, color=CURVE_COLOR, linewidth=2)
    ax2.scatter([offset], [val], color=CURVE_COLOR, s=45, zorder=5)
    ax2.set_xlim(OFFSETS[0], OFFSETS[-1])
    ax2.set_ylim(0, 1.0)
    ax2.set_xlabel("horizontal offset between box centers", fontsize=9)
    ax2.set_ylabel("IoU = intersection / union", fontsize=9)
    ax2.axhline(0, color="#bbbbbb", linewidth=0.8)
    ax2.tick_params(labelsize=8)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    fig.suptitle(f"intersection over union (IoU)   —   IoU = {val:.2f}",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
for i, off in enumerate(OFFSETS):
    hold = 10
    if abs(off) < 1e-6:
        hold = 35  # pause at full overlap, IoU=1
    frames += [render(off)] * 1
    if hold > 1:
        frames += [render(off, freeze_hold=True)] * (hold - 1)
frames += [render(OFFSETS[-1], freeze_hold=True)] * 60  # final hold on the completed curve

frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=50, loop=0)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
