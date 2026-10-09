"""Generate floodfill_demo.gif: an animated walkthrough of the stack-based
flood fill algorithm from the lesson 4 notebook (flood_fill_stack), growing
outward from a seed pixel.

Uses a massively downsampled version of the same binary "fruit" image and
seed point as the notebook's flood-fill cell, so individual pixels are large
enough to watch the stack's frontier grow one batch of pops at a time.

Run from anywhere:
    python make_floodfill_demo.py
Output:
    floodfill_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "floodfill_demo.gif")
IMG_DIR = os.path.join(OUT_DIR, "..", "img")

# Same binarization pipeline and seed as the lesson 4 notebook.
img = cv2.imread(os.path.join(IMG_DIR, "fruit.jpg"), cv2.IMREAD_GRAYSCALE)
_, im_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
im_bin2 = cv2.morphologyEx(im_bin, cv2.MORPH_OPEN, kernel)
SEED_ORIG = (60, 110)  # (x, y) inside the first banana, same as the notebook

# Downsample drastically so each pixel renders as a large, countable square.
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
seed = (round(SEED_ORIG[0] * scale), round(SEED_ORIG[1] * scale))
assert binary[seed[1], seed[0]] != 0, "seed must land on foreground after downsampling"

BG_COLOR = (40, 40, 40)
FG_COLOR = (255, 255, 255)
FRONTIER_COLOR = (224, 123, 0)
FILLED_COLOR = (58, 111, 176)
MATCH_COLOR = "#2e9e44"
ACTIVE_COLOR = "#e07b00"
FILLED_COLOR_HEX = "#3a6fb0"


def flood_fill_steps(binary, seed, new_fills_per_step):
    """Same algorithm as flood_fill_stack in the notebook, but snapshotting
    (filled mask, frontier set) every time `new_fills_per_step` *new* pixels
    get filled -- not every N pops -- so the animation advances at a
    constant visual pace even though many pops are stale/invalid and don't
    actually fill anything (duplicate stack entries, already-filled pixels)."""
    filled = np.zeros_like(binary, dtype=bool)
    h, w = binary.shape
    stack = [seed]
    steps = [(filled.copy(), set(stack))]
    while stack:
        newly_filled = 0
        while stack and newly_filled < new_fills_per_step:
            x, y = stack.pop()
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            if filled[y, x] or binary[y, x] == 0:
                continue
            filled[y, x] = True
            newly_filled += 1
            stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
        steps.append((filled.copy(), set(stack)))
    return steps


NEW_FILLS_PER_STEP = 2
steps = flood_fill_steps(binary, seed, NEW_FILLS_PER_STEP)
TOTAL_FG = int((binary > 0).sum())


def render(step_i, final_hold=False):
    filled, frontier = steps[step_i]

    color_img = np.empty((H, W, 3), dtype=np.uint8)
    color_img[:] = BG_COLOR
    color_img[binary > 0] = FG_COLOR
    color_img[filled] = FILLED_COLOR
    # The stack can hold stale duplicate entries for pixels already filled via
    # another path before being popped -- skip those so they don't overwrite
    # the filled color with the frontier color.
    for (x, y) in frontier:
        if 0 <= x < W and 0 <= y < H and not filled[y, x] and binary[y, x] != 0:
            color_img[y, x] = FRONTIER_COLOR

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=120)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.1)
    ax.imshow(color_img, interpolation="nearest")
    ax.scatter(*seed, c="red", s=50, marker="x", linewidths=2, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(0.13, 0.045, "✕", color="red", fontsize=12, ha="center", va="center", fontweight="bold")
    fig.text(0.16, 0.045, "seed pixel", color="#333333", fontsize=9.5, ha="left", va="center")
    fig.text(0.42, 0.045, "■", color=FILLED_COLOR_HEX, fontsize=15, ha="center", va="center")
    fig.text(0.46, 0.045, "filled pixels", color="#333333", fontsize=9.5, ha="left", va="center")
    fig.text(0.70, 0.045, "■", color=ACTIVE_COLOR, fontsize=15, ha="center", va="center")
    fig.text(0.74, 0.045, "frontier pixels (stack)", color="#333333", fontsize=9.5, ha="left", va="center")

    filled_count = int(filled.sum())
    true_frontier = sum(1 for (x, y) in frontier
                          if 0 <= x < W and 0 <= y < H and not filled[y, x] and binary[y, x] != 0)
    done = step_i == len(steps) - 1
    if final_hold:
        suptitle = f"complete: {filled_count} of {TOTAL_FG} foreground pixels filled, frontier empty"
        color = MATCH_COLOR
    elif done:
        suptitle = f"filled = {filled_count} / {TOTAL_FG}, frontier empty — stack exhausted"
        color = "#222222"
    else:
        suptitle = f"filled = {filled_count} / {TOTAL_FG},  frontier (stack) = {true_frontier} pixels"
        color = ACTIVE_COLOR
    fig.suptitle(suptitle, fontsize=12.5, color=color,
                  fontweight="bold" if final_hold else "normal")
    fig.text(0.5, 0.9, "flood_fill_stack(binary, seed): pop a pixel, fill it, push its 4 neighbors",
               ha="center", va="top", fontsize=9.5, color="#555555")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_START = 10
HOLD_STEP = 3
HOLD_DONE = 14
HOLD_END = 40

frames += [render(0)] * HOLD_START
for i in range(1, len(steps)):
    frames += [render(i)] * HOLD_STEP
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
      f"{int((binary > 0).sum())} fg pixels, seed={seed})")
