"""Generate canny_stages_demo.gif: an animated walkthrough of the lesson 12
notebook's 4-stage Canny pipeline description (cell b8293efa): smooth ->
gradient magnitude -> non-maximum suppression -> hysteresis. The first three
stages are whole-image transforms shown as a single reveal each; hysteresis
is a genuine propagation process (strong edges "recruit" connected weak
pixels), so it's animated step by step like the lesson's flood-fill demo.

Uses a small crop of the same calvin.png image as the notebook's Canny cell,
since cv2.Canny doesn't expose its intermediate stages -- NMS and hysteresis
are implemented from scratch here, matching the textbook algorithm.

Run from anywhere:
    python make_canny_stages_demo.py
Output:
    canny_stages_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "canny_stages_demo.gif")
IMG_DIR = os.path.join(OUT_DIR, "..", "img")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"
STRONG_COLOR = (255, 255, 255)
WEAK_LINKED_COLOR = (224, 123, 0)
DISCARDED_COLOR = (60, 60, 90)
BG_COLOR = (15, 15, 15)

HIGH_THRESH = 55.0
LOW_THRESH = 11.0

calvin = cv2.imread(os.path.join(IMG_DIR, "calvin.png"), cv2.IMREAD_GRAYSCALE)
crop = calvin[20:140, 130:250]
H, W = crop.shape

# A larger sigma smooths out more noise before differentiating, which makes
# the gradient direction vary more smoothly from pixel to pixel -- that in
# turn makes non-max suppression's 4-direction quantization less prone to
# "jittering" off the true ridge and breaking edge connectivity.
smooth = cv2.GaussianBlur(crop, (9, 9), 2.2)

gx = cv2.Sobel(smooth.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(smooth.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
mag = np.sqrt(gx ** 2 + gy ** 2)

# --- non-maximum suppression: compare each pixel's magnitude to the exact
# (bilinearly-interpolated) magnitude one step forward and backward along
# the true gradient direction, rather than snapping to one of 4 discrete
# directions. Snapping to 4 bins can suppress a pixel that should survive
# wherever a curving edge crosses a bin boundary, breaking connectivity --
# this sub-pixel version avoids that artifact.
norm = np.maximum(mag, 1e-6)
cos_t, sin_t = gx / norm, gy / norm


def bilinear_at(img, x, y):
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    if x0 < 0 or x0 + 1 >= W or y0 < 0 or y0 + 1 >= H:
        return 0.0
    dx, dy = x - x0, y - y0
    return ((1 - dx) * (1 - dy) * img[y0, x0] + dx * (1 - dy) * img[y0, x0 + 1] +
            (1 - dx) * dy * img[y0 + 1, x0] + dx * dy * img[y0 + 1, x0 + 1])


nms = np.zeros_like(mag)
for y in range(H):
    for x in range(W):
        if mag[y, x] == 0:
            continue
        c, s = cos_t[y, x], sin_t[y, x]
        m_fwd = bilinear_at(mag, x + c, y + s)
        m_bwd = bilinear_at(mag, x - c, y - s)
        if mag[y, x] >= m_fwd and mag[y, x] >= m_bwd:
            nms[y, x] = mag[y, x]

# --- hysteresis setup: strong/weak classification, then BFS from every
# strong pixel through 8-connected weak neighbors (animated below).
strong = nms > HIGH_THRESH
weak = (nms > LOW_THRESH) & ~strong


def hysteresis_steps(new_links_per_step):
    edges = strong.copy()
    frontier = set(zip(*np.nonzero(strong)[::-1]))  # (x, y) pairs
    stack = [(x, y) for (x, y) in frontier]
    steps = [(edges.copy(), set(stack))]
    while stack:
        newly_linked = 0
        while stack and newly_linked < new_links_per_step:
            x, y = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H and weak[ny, nx] and not edges[ny, nx]:
                        edges[ny, nx] = True
                        newly_linked += 1
                        stack.append((nx, ny))
        steps.append((edges.copy(), set(stack)))
    return steps


HYSTERESIS_STEPS = hysteresis_steps(new_links_per_step=8)


# Shared margins for every frame (stage panels and the hysteresis panel
# alike) so the image stays the same size and position throughout -- the
# bottom strip is reserved for the legend even on frames that don't draw one.
PANEL_RECT = dict(top=0.90, bottom=0.14, left=0.03, right=0.97)


def gray_panel(img, vmax=None):
    fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=120)
    fig.subplots_adjust(**PANEL_RECT)
    ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax


def render_stage(stage):
    """stage: 0=original, 1=smooth, 2=gradient magnitude, 3=NMS."""
    titles = ["original", "smooth (Gaussian blur)", "gradient magnitude", "non-max suppression"]
    imgs = [crop, smooth, mag, nms]
    # NMS zeroes out most pixels, leaving a sparse ridge whose surviving
    # values span a huge range (a few very bright peaks, lots of dim-but-real
    # ridge pixels) -- normalizing to mag.max() crushes those dim pixels to
    # near-black, making a mostly-continuous edge look broken into dashes.
    # Normalize to a lower percentile instead so the whole ridge is visible.
    vmaxes = [255, 255, mag.max(), HIGH_THRESH * 1.3]
    fig, ax = gray_panel(imgs[stage], vmax=vmaxes[stage])
    fig.suptitle(titles[stage], fontsize=13, color=ACTIVE_COLOR if stage else NEUTRAL_COLOR,
                  fontweight="bold")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_hysteresis(step_i, final_hold=False):
    edges, frontier = HYSTERESIS_STEPS[step_i]
    color_img = np.empty((H, W, 3), dtype=np.uint8)
    color_img[:] = BG_COLOR
    color_img[weak] = DISCARDED_COLOR
    if final_hold:
        # final result: strong and linked-weak pixels are both just "edge" now
        color_img[edges] = STRONG_COLOR
    else:
        color_img[edges & ~strong] = WEAK_LINKED_COLOR
        color_img[strong] = STRONG_COLOR

    fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=120)
    fig.subplots_adjust(**PANEL_RECT)
    ax.imshow(color_img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    def swatch(x, y, color, label, bordered=False):
        if bordered:
            fig.add_artist(Rectangle((x - 0.012, y - 0.012), 0.024, 0.024, facecolor=color,
                                       edgecolor="#333333", linewidth=0.8, transform=fig.transFigure))
        else:
            fig.text(x, y, "■", color=color, fontsize=13, ha="center", va="center")
        fig.text(x + 0.04, y, label, color="#333333", fontsize=9, ha="left", va="center")

    if final_hold:
        swatch(0.30, 0.06, "white", "edge", bordered=True)
        swatch(0.56, 0.06, "#4a4a6a", "discarded")
    else:
        swatch(0.18, 0.06, "white", "strong", bordered=True)
        swatch(0.42, 0.06, "#e07b00", "linked weak")
        swatch(0.70, 0.06, "#4a4a6a", "discarded weak")

    if final_hold:
        suptitle = "hysteresis complete"
        color = MATCH_COLOR
    else:
        suptitle = "hysteresis: strong edges recruit connected weak pixels"
        color = ACTIVE_COLOR
    fig.suptitle(suptitle, fontsize=11.5, color=color, fontweight="bold" if final_hold else "normal")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_canny_result():
    """The same final edge set as the 'hysteresis complete' frame (strong +
    linked-weak pixels, all shown as plain white) -- just without the
    strong/linked-weak color distinction, as the concluding "Canny result"."""
    edges, _ = HYSTERESIS_STEPS[-1]
    color_img = np.empty((H, W, 3), dtype=np.uint8)
    color_img[:] = BG_COLOR
    color_img[edges] = STRONG_COLOR

    fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=120)
    fig.subplots_adjust(**PANEL_RECT)
    ax.imshow(color_img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.add_artist(Rectangle((0.5 - 0.012, 0.06 - 0.012), 0.024, 0.024, facecolor="white",
                               edgecolor="#333333", linewidth=0.8, transform=fig.transFigure))
    fig.text(0.5 + 0.04, 0.06, "edge", color="#333333", fontsize=9, ha="left", va="center")

    fig.suptitle("Canny result", fontsize=13, color=MATCH_COLOR, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STAGE = 22
HOLD_STEP = 3
HOLD_DONE = 16
HOLD_END = 44

for stage in range(4):
    frames += [render_stage(stage)] * HOLD_STAGE

for i in range(1, len(HYSTERESIS_STEPS)):
    frames += [render_hysteresis(i)] * HOLD_STEP
frames += [render_hysteresis(len(HYSTERESIS_STEPS) - 1)] * HOLD_DONE
frames += [render_hysteresis(len(HYSTERESIS_STEPS) - 1, final_hold=True)] * HOLD_DONE
frames += [render_canny_result()] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=90,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames, {len(HYSTERESIS_STEPS)} hysteresis steps)")
