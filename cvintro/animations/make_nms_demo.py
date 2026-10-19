"""Generate nms_demo.gif: an animated companion to the lesson 40 notebook's
"Step 3: threshold, then merge duplicates" section. Animates the
non-maximum-suppression loop itself, round by round: pick the
highest-scoring remaining detection, suppress everything that overlaps it
too much, repeat -- ending on the same 2 final boxes as the book.

Reuses the notebook's own make_face/make_nonface/WindowClassifier/
make_scene/sliding_window_scores/iou/nms setup verbatim (data rng seed=11,
model seed=0, scene rng seed=21, threshold = 97.5th percentile, iou_thresh
=0.3), so the final boxes match the book exactly.

Run from anywhere:
    python make_nms_demo.py
Output:
    nms_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "nms_demo.gif")

TEXT_COLOR = "#222222"
RAW_COLOR = "#f2994a"      # not-yet-decided raw detections
BEST_COLOR = "#d62728"     # the detection currently being picked (matches the
                           # red-dashed "predicted" convention of the static figures)
SUPPRESS_COLOR = "#9e9e9e"  # detections just suppressed this round
TRUE_COLOR = "#2e9e44"


def make_face(size=16, rng=None):
    img = np.zeros((size, size), dtype=np.float32)
    cy, cx = size // 2, size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    img[((xx - cx) ** 2 + (yy - cy) ** 2) <= (size * 0.42) ** 2] = 0.6
    img[(np.abs(xx - (cx - 3)) <= 1) & (np.abs(yy - (cy - 2)) <= 1)] = 1.0
    img[(np.abs(xx - (cx + 3)) <= 1) & (np.abs(yy - (cy - 2)) <= 1)] = 1.0
    img[(np.abs(xx - cx) <= 2) & (np.abs(yy - (cy + 3)) <= 1)] = 0.9
    if rng is not None:
        img = np.clip(img + rng.normal(0, 0.08, img.shape), 0, 1).astype(np.float32)
    return img


def make_nonface(size=16, rng=None):
    img = rng.uniform(0, 0.5, (size, size)).astype(np.float32)
    if rng.random() < 0.5:
        cy, cx = rng.integers(2, size - 2), rng.integers(2, size - 2)
        r = rng.integers(2, 5)
        yy, xx = np.mgrid[0:size, 0:size]
        img[((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2] = rng.uniform(0.4, 0.9)
    return img


rng = np.random.default_rng(11)
N = 200
faces = np.array([make_face(rng=rng) for _ in range(N)])
nonfaces = np.array([make_nonface(rng=rng) for _ in range(N)])
X = np.concatenate([faces, nonfaces])
y = np.concatenate([np.ones(N), np.zeros(N)]).astype(np.float32)
perm = rng.permutation(len(X))
X, y = X[perm], y[perm]
split = int(0.8 * len(X))
Xtr, ytr = X[:split], y[:split]


class WindowClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool2d(1),
        )
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1)).squeeze(-1)


print("training window classifier...")
torch.manual_seed(0)
model = WindowClassifier()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
Xt = torch.tensor(Xtr).unsqueeze(1); yt = torch.tensor(ytr)
for _ in range(300):
    opt.zero_grad()
    loss = F.binary_cross_entropy_with_logits(model(Xt), yt)
    loss.backward()
    opt.step()


def make_scene(rng, size=64, face_size=16, n_faces=2):
    scene = rng.uniform(0, 0.5, (size, size)).astype(np.float32)
    corners = []
    tries = 0
    while len(corners) < n_faces and tries < 50:
        tries += 1
        x0 = rng.integers(0, size - face_size)
        y0 = rng.integers(0, size - face_size)
        if any(abs(x0 - px) < face_size and abs(y0 - py) < face_size for px, py in corners):
            continue
        face = make_face(size=face_size, rng=rng)
        scene[y0:y0 + face_size, x0:x0 + face_size] = np.maximum(scene[y0:y0 + face_size, x0:x0 + face_size], face)
        corners.append((x0, y0))
    centers = [(float(x0 + face_size / 2), float(y0 + face_size / 2)) for x0, y0 in corners]
    return scene, centers


scene_rng = np.random.default_rng(21)
scene, true_boxes = make_scene(scene_rng)


def sliding_window_scores(model, scene, win=16, stride=1):
    half = win // 2
    scores = np.full(scene.shape, np.nan, dtype=np.float32)
    with torch.no_grad():
        for y0 in range(0, scene.shape[0] - win + 1, stride):
            for x0 in range(0, scene.shape[1] - win + 1, stride):
                patch = scene[y0:y0 + win, x0:x0 + win]
                scores[y0 + half, x0 + half] = model(torch.tensor(patch[None, None]).float()).item()
    return scores


print("computing the score map...")
score_map = sliding_window_scores(model, scene)


def iou(a, b):
    acx, acy, aw, ah = a[:4]; bcx, bcy, bw, bh = b[:4]
    ax0, ay0, ax1, ay1 = acx - aw / 2, acy - ah / 2, acx + aw / 2, acy + ah / 2
    bx0, by0, bx1, by1 = bcx - bw / 2, bcy - bh / 2, bcx + bw / 2, bcy + bh / 2
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


IOU_THRESH = 0.3
valid = ~np.isnan(score_map)
threshold = np.percentile(score_map[valid], 97.5)
raw_detections = [(cx, cy, 16, 16, score_map[cy, cx])
                   for cy, cx in zip(*np.where(valid)) if score_map[cy, cx] > threshold]
print(f"threshold: {threshold:.2f}, raw detections: {len(raw_detections)}")

# ---------------------------------------------------------------------------
# Trace the NMS loop, recording one state per animation "beat" so the
# rendering code below never has to re-derive the algorithm's logic.
# ---------------------------------------------------------------------------
dets = sorted(raw_detections, key=lambda d: -d[4])
beats = [dict(phase="intro", remaining=list(dets), kept=[], best=None, suppressed=[])]
remaining = list(dets)
kept = []
round_idx = 0
while remaining:
    round_idx += 1
    best = remaining[0]
    others = remaining[1:]
    beats.append(dict(phase="select", round=round_idx, remaining=list(remaining), kept=list(kept),
                       best=best, suppressed=[]))
    suppressed = [d for d in others if iou(best, d) >= IOU_THRESH]
    survivors = [d for d in others if iou(best, d) < IOU_THRESH]
    beats.append(dict(phase="suppress", round=round_idx, remaining=list(remaining), kept=list(kept),
                       best=best, suppressed=suppressed))
    kept.append(best)
    remaining = survivors
    beats.append(dict(phase="after", round=round_idx, remaining=list(remaining), kept=list(kept),
                       best=None, suppressed=suppressed))
final_detections = kept
beats.append(dict(phase="final", remaining=[], kept=list(kept), best=None, suppressed=[]))
print(f"NMS rounds: {round_idx}, final detections: {len(final_detections)}")


def draw_box(ax, d, color, lw=1.6, ls="-"):
    cx, cy, w, h, s = d
    ax.add_patch(patches.Rectangle((cx - w / 2, cy - h / 2), w, h, edgecolor=color,
                                    facecolor="none", linewidth=lw, linestyle=ls))


PICK_LINE = "pick the highest-scoring remaining detection"


def suppress_line(b):
    return f"suppress overlaps (IoU ≥ {IOU_THRESH})  — {len(b['suppressed'])} removed"


def done_line(b):
    return f"done: {len(b['kept'])} kept so far, {len(b['remaining'])} remaining"


BLANK = " "

# Three fixed text slots per beat, so the block never changes height: line 1
# ("pick...") appears first and stays, line 2 ("suppress...") joins it, then
# line 3 ("done...") joins both -- all three then clear together once the
# next round's "select" beat starts. The round number itself is shown once,
# off to the side, rather than repeated in every line.
ROUND_LABEL = {
    "intro": lambda b: "",
    "select": lambda b: f"round {b['round']}",
    "suppress": lambda b: f"round {b['round']}",
    "after": lambda b: f"round {b['round']}",
    "final": lambda b: "",
}
LINES = {
    "intro": lambda b: (f"{len(b['remaining'])} raw detections above threshold ({threshold:.2f})",
                         BLANK, BLANK),
    "select": lambda b: (PICK_LINE, BLANK, BLANK),
    "suppress": lambda b: (PICK_LINE, suppress_line(b), BLANK),
    "after": lambda b: (PICK_LINE, suppress_line(b), done_line(b)),
    "final": lambda b: (f"NMS result: {len(b['kept'])} detections vs. {len(true_boxes)} true faces",
                         BLANK, BLANK),
}


def render(beat):
    fig, ax = plt.subplots(figsize=(5.4, 6.0), dpi=120)
    fig.subplots_adjust(top=0.78, bottom=0.03, left=0.03, right=0.97)

    ax.imshow(scene, cmap="gray", vmin=0, vmax=1)

    show_true = beat["phase"] == "final"
    if show_true:
        for cx, cy in true_boxes:
            ax.add_patch(patches.Rectangle((cx - 8, cy - 8), 16, 16, edgecolor=TRUE_COLOR,
                                            facecolor="none", linewidth=3.0))

    # Kept detections are always drawn red dashed, same as the static result
    # figure above this animation -- in the final beat that's on top of the
    # solid green ground-truth boxes, so green shows through the gaps.
    for d in beat["kept"]:
        draw_box(ax, d, BEST_COLOR, lw=1.8, ls="--")
    if beat["phase"] != "final":
        pending = beat["remaining"]
        if beat["best"] is not None:
            pending = [d for d in pending if d is not beat["best"]]
        for d in pending:
            draw_box(ax, d, RAW_COLOR, lw=1.1, ls=":")
    if beat["phase"] == "suppress":
        for d in beat["suppressed"]:
            draw_box(ax, d, SUPPRESS_COLOR, lw=1.8, ls=":")
    if beat["best"] is not None:
        draw_box(ax, beat["best"], BEST_COLOR, lw=2.8, ls="--")

    ax.axis("off")
    phase = beat["phase"]
    line1, line2, line3 = LINES[phase](beat)
    ax.set_title(f"{line1}\n{line2}\n{line3}", fontsize=10.5, color=TEXT_COLOR, fontweight="bold")
    fig.text(0.5, 0.905, ROUND_LABEL[phase](beat), fontsize=10.5, color="#777777",
              fontweight="bold", ha="center", va="top")
    fig.suptitle("non-maximum suppression", fontsize=13, fontweight="bold", color=TEXT_COLOR, y=0.99)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


HOLD = {"intro": 35, "select": 32, "suppress": 32, "after": 34, "final": 90}

frames = []
for beat in beats:
    frames += [render(beat)] * HOLD[beat["phase"]]

frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=50, loop=0)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
