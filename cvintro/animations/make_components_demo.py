"""Generate components_demo.gif: an animated walkthrough of a simplified
connected-components algorithm (scan the image; whenever an unlabeled
foreground pixel is found, flood-fill that whole blob with a new label).

This is NOT the two-pass union-find algorithm that cv2.connectedComponentsWithStats
actually uses internally -- it's a simpler scan-and-flood-fill approach that's
easier to visualize but produces the same labeling.

Uses the same (massively downsampled) binary "fruit" image as the lesson 4
notebook's flood-fill animation, so the two animations are visually consistent.

Run from anywhere:
    python make_components_demo.py
Output:
    components_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "components_demo.gif")
IMG_DIR = os.path.join(OUT_DIR, "..", "img")

# Same binarization pipeline as the lesson 4 notebook.
img = cv2.imread(os.path.join(IMG_DIR, "fruit.jpg"), cv2.IMREAD_GRAYSCALE)
_, im_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
im_bin2 = cv2.morphologyEx(im_bin, cv2.MORPH_OPEN, kernel)

# Same downsampling as the flood-fill animation, for visual consistency.
# Dilate first so thin parts (e.g. a banana's stem tip) survive the
# area-average downsampling instead of being thresholded away into a
# disconnected speck.
im_bin2_dilated = cv2.dilate(im_bin2, np.ones((6, 6), np.uint8))
TARGET_W = 36
scale = TARGET_W / im_bin2.shape[1]
TARGET_H = round(im_bin2.shape[0] * scale)
small = cv2.resize(im_bin2_dilated, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
_, binary = cv2.threshold(small, 127, 255, cv2.THRESH_BINARY)
H, W = binary.shape

BG_COLOR = (40, 40, 40)
FG_COLOR = (255, 255, 255)
FRONTIER_COLOR = (224, 123, 0)
ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"

# Same palette (minus black, which is reserved for background) as the
# notebook's connected-components visualization cell.
LABEL_COLORS = [
    (220, 40, 40),
    (0, 170, 90),
    (40, 90, 220),
    (200, 170, 0),
    (0, 170, 170),
    (170, 0, 170),
]


def label_color(label):
    return LABEL_COLORS[(label - 1) % len(LABEL_COLORS)]


def component_steps(binary, new_fills_per_step):
    """Scan the image in raster order. Whenever an unlabeled foreground pixel
    is found, flood-fill that whole blob with a new label, snapshotting
    (labels, frontier, current_label, seed) every `new_fills_per_step` new
    fills, just like the flood-fill animation, so the pace stays constant."""
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=int)
    steps = [(labels.copy(), set(), 0, None)]
    current_label = 0
    for y in range(h):
        for x in range(w):
            if binary[y, x] == 0 or labels[y, x] != 0:
                continue
            current_label += 1
            stack = [(x, y)]
            steps.append((labels.copy(), {(x, y)}, current_label, (x, y)))
            while stack:
                newly_filled = 0
                while stack and newly_filled < new_fills_per_step:
                    px, py = stack.pop()
                    if px < 0 or px >= w or py < 0 or py >= h:
                        continue
                    if labels[py, px] != 0 or binary[py, px] == 0:
                        continue
                    labels[py, px] = current_label
                    newly_filled += 1
                    stack.extend([(px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)])
                steps.append((labels.copy(), set(stack), current_label, (x, y)))
    steps.append((labels.copy(), set(), current_label, None))
    return steps, current_label


NEW_FILLS_PER_STEP = 2
steps, NUM_LABELS = component_steps(binary, NEW_FILLS_PER_STEP)
TOTAL_FG = int((binary > 0).sum())


def render(step_i, final_hold=False):
    labels, frontier, current_label, seed = steps[step_i]

    color_img = np.empty((H, W, 3), dtype=np.uint8)
    color_img[:] = BG_COLOR
    color_img[binary > 0] = FG_COLOR
    for lbl in range(1, current_label + 1):
        color_img[labels == lbl] = label_color(lbl)
    for (x, y) in frontier:
        if 0 <= x < W and 0 <= y < H and labels[y, x] == 0 and binary[y, x] != 0:
            color_img[y, x] = FRONTIER_COLOR

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=120)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.1)
    ax.imshow(color_img, interpolation="nearest")
    if seed is not None:
        ax.scatter(*seed, c="red", s=50, marker="x", linewidths=2, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_color = label_color(current_label) if current_label > 0 else (68, 68, 68)
    legend_color_hex = "#%02x%02x%02x" % legend_color

    fig.text(0.13, 0.045, "✕", color="red", fontsize=12, ha="center", va="center", fontweight="bold")
    fig.text(0.16, 0.045, "new seed found", color="#333333", fontsize=9.5, ha="left", va="center")
    fig.text(0.46, 0.045, "■", color=legend_color_hex, fontsize=15, ha="center", va="center")
    fig.text(0.50, 0.045, "labeled component", color="#333333", fontsize=9.5, ha="left", va="center")
    fig.text(0.78, 0.045, "■", color=ACTIVE_COLOR, fontsize=15, ha="center", va="center")
    fig.text(0.82, 0.045, "frontier (stack)", color="#333333", fontsize=9.5, ha="left", va="center")

    labeled_count = int((labels > 0).sum())
    done = step_i == len(steps) - 1
    if final_hold:
        suptitle = f"complete: found {current_label} connected components"
        color = MATCH_COLOR
    elif done:
        suptitle = f"complete: found {current_label} connected components"
        color = "#222222"
    elif seed is not None and labeled_count == 0:
        suptitle = f"scanning... found a new unlabeled pixel, starting component {current_label}"
        color = ACTIVE_COLOR
    else:
        suptitle = f"growing component {current_label}: {labeled_count} / {TOTAL_FG} foreground pixels labeled so far"
        color = ACTIVE_COLOR
    fig.suptitle(suptitle, fontsize=12.5, color=color,
                  fontweight="bold" if final_hold else "normal")
    fig.text(0.5, 0.9, "scan the image; flood-fill each new unlabeled blob with its own label",
               ha="center", va="top", fontsize=9.5, color="#555555")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_START = 10
HOLD_STEP = 3
HOLD_SEED = 10
HOLD_DONE = 14
HOLD_END = 40

frames += [render(0)] * HOLD_START
prev_label = 0
for i in range(1, len(steps)):
    _, _, current_label, seed = steps[i]
    is_new_seed = seed is not None and current_label != prev_label
    prev_label = current_label
    frames += [render(i)] * (HOLD_SEED if is_new_seed else HOLD_STEP)
frames += [render(len(steps) - 1)] * HOLD_DONE
frames += [render(len(steps) - 1, final_hold=True)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=90,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames, {len(steps)} steps, "
      f"{NUM_LABELS} components, {TOTAL_FG} fg pixels)")
