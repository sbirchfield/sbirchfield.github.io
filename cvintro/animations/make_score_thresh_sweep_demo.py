"""Generate score_thresh_sweep_demo.gif: an animated companion to the
lesson 41 notebook's "In practice: real detectors on real images" section.
Runs the pretrained Faster R-CNN once on the COCO sample photo, then sweeps
score_thresh down from high to low, revealing which boxes survive at each
cutoff -- including the low-confidence "couch"/"bed" guesses the book's
score_thresh=0.7 cutoff hides.

Reuses the notebook's own torchvision Faster R-CNN / COCO sample image
setup verbatim (weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1, eval mode,
no training), so the detections and scores match the book exactly.

Run from anywhere:
    python make_score_thresh_sweep_demo.py
Output:
    score_thresh_sweep_demo.gif  (written next to this script)
"""
import io
import os
import urllib.request
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "score_thresh_sweep_demo.gif")

TEXT_COLOR = "#222222"
BOX_COLOR = "#2e9e44"

CACHE_DIR = Path.home() / ".cache" / "cvintro"
COCO_IMG_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
COCO_IMG_PATH = CACHE_DIR / "coco_sample.jpg"


def ensure_coco_sample():
    if COCO_IMG_PATH.exists():
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading a COCO val2017 sample image...")
    urllib.request.urlretrieve(COCO_IMG_URL, COCO_IMG_PATH)


ensure_coco_sample()

print("loading pretrained Faster R-CNN (COCO)...")
weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
real_detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
real_detector.eval()
coco_classes = weights.meta["categories"]

img_bgr = cv2.imread(str(COCO_IMG_PATH))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_tensor = torch.tensor(img_rgb / 255.0, dtype=torch.float32).permute(2, 0, 1)

print("running one forward pass...")
with torch.no_grad():
    result = real_detector([img_tensor])[0]

boxes = result["boxes"].numpy()
labels = result["labels"].numpy()
scores = result["scores"].numpy()
order = np.argsort(-scores)
boxes, labels, scores = boxes[order], labels[order], scores[order]
print(f"{len(scores)} raw candidate boxes, top score {scores[0]:.2f}, "
      f"{int((scores > 0.7).sum())} above 0.7, {int((scores > 0.3).sum())} above 0.3")

THRESHOLDS = list(np.arange(0.95, 0.30, -0.025)) + [0.3] + list(np.arange(0.30, 0.049, -0.025))
BOOK_THRESH = 0.7
EXERCISE_THRESH = 0.3

SLIDER_MIN, SLIDER_MAX = 0.0, 1.0
SLIDER_COLOR = "#bbbbbb"
HANDLE_COLOR = "#1f6fb4"
TICK_COLOR = "#2e9e44"

# Static reference marks, drawn every frame at the same spot regardless of
# where the moving handle currently is: the book's fixed cutoff (described
# in the text just above this animation) and the lower cutoff explored in
# the exercises below.
TICKS = [(BOOK_THRESH, "recommended cutoff"), (EXERCISE_THRESH, "exercise cutoff")]


def draw_slider(fig, thresh):
    """A horizontal slider spanning score_thresh in [0, 1], with a moving
    handle at the current threshold and fixed tick marks for the two
    cutoffs referenced elsewhere in the notebook."""
    sax = fig.add_axes([0.14, 0.015, 0.72, 0.1])
    sax.set_xlim(SLIDER_MIN, SLIDER_MAX)
    sax.set_ylim(-1.1, 1.6)
    sax.axis("off")

    sax.text(SLIDER_MIN, 1.3, "0.0", ha="left", va="bottom", fontsize=8, color=TEXT_COLOR)
    sax.text(0.5, 1.3, "score_thresh", ha="center", va="bottom", fontsize=8.5,
              color=TEXT_COLOR, fontweight="bold")
    sax.text(SLIDER_MAX, 1.3, "1.0", ha="right", va="bottom", fontsize=8, color=TEXT_COLOR)

    sax.plot([SLIDER_MIN, SLIDER_MAX], [0.5, 0.5], color=SLIDER_COLOR, linewidth=5,
              solid_capstyle="round", zorder=1)

    for pos, label in TICKS:
        sax.plot([pos, pos], [-0.1, 1.1], color=TICK_COLOR, linewidth=2.2, zorder=2)
        sax.text(pos, -0.4, label, ha="center", va="top", fontsize=7.6,
                  color=TICK_COLOR, fontweight="bold")

    sax.scatter([thresh], [0.5], s=140, color=HANDLE_COLOR, edgecolor="white",
                linewidth=1.4, zorder=3)


def render(thresh):
    keep = scores > thresh
    n = int(keep.sum())

    fig, ax = plt.subplots(figsize=(7.4, 6.3), dpi=120)
    fig.subplots_adjust(top=0.90, bottom=0.17, left=0.02, right=0.98)

    ax.imshow(img_rgb)
    for box, label, score in zip(boxes[keep], labels[keep], scores[keep]):
        x0, y0, x1, y1 = box
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor=BOX_COLOR,
                                        facecolor="none", linewidth=2))
        ax.text(x0, max(y0 - 5, 10), f"{coco_classes[label]} {score:.2f}", color=BOX_COLOR,
                fontsize=9, weight="bold")
    ax.axis("off")

    ax.set_title(f"score_thresh = {thresh:.2f}   —   {n} detections", fontsize=12.5,
                 fontweight="bold", color=TEXT_COLOR, pad=12)

    draw_slider(fig, thresh)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
for t in THRESHOLDS:
    hold = 8
    if abs(t - BOOK_THRESH) < 1e-6 or abs(t - EXERCISE_THRESH) < 1e-6:
        hold = 55
    frames += [render(t)] * hold
frames += [render(THRESHOLDS[-1])] * 70

frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=55, loop=0)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
