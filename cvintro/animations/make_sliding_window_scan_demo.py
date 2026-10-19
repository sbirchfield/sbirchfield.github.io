"""Generate sliding_window_scan_demo.gif: an animated companion to the
lesson 40 notebook's "Step 2: slide the window across a scene" section.
Animates the window classifier physically sliding across the scene,
revealing the score heatmap row by row as it goes, ending matched to the
book's score-map figure.

Reuses the notebook's own make_face/make_nonface/WindowClassifier/
make_scene/sliding_window_scores setup verbatim (data rng seed=11, model
seed=0, scene rng seed=21), so the final heatmap matches the book exactly.

Run from anywhere:
    python make_sliding_window_scan_demo.py
Output:
    sliding_window_scan_demo.gif  (written next to this script)
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
OUT_GIF = os.path.join(OUT_DIR, "sliding_window_scan_demo.gif")

TEXT_COLOR = "#222222"
BOX_COLOR = "#f2c200"


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

WIN, STRIDE = 16, 1
HALF = WIN // 2
y0_range = list(range(0, scene.shape[0] - WIN + 1, STRIDE))
x0_range = list(range(0, scene.shape[1] - WIN + 1, STRIDE))

print("computing the full score map...")
score_map = np.full(scene.shape, np.nan, dtype=np.float32)
with torch.no_grad():
    for y0 in y0_range:
        for x0 in x0_range:
            patch = scene[y0:y0 + WIN, x0:x0 + WIN]
            score_map[y0 + HALF, x0 + HALF] = model(torch.tensor(patch[None, None]).float()).item()

cmap = plt.cm.hot.copy()
cmap.set_bad(color="gray")


def render(revealed_rows, cur_y0, cur_x0, scanning):
    """revealed_rows: set of y0 values whose full row is already revealed.
    cur_y0/cur_x0: the window currently being evaluated (its top-left corner).
    scanning: False only for the final held frame (hides the scanning box)."""
    display = np.full(scene.shape, np.nan, dtype=np.float32)
    for y0 in revealed_rows:
        display[y0 + HALF, :] = score_map[y0 + HALF, :]
    # partially reveal the current row up through cur_x0
    if cur_y0 is not None:
        for x0 in x0_range:
            if x0 <= cur_x0:
                display[cur_y0 + HALF, x0 + HALF] = score_map[cur_y0 + HALF, x0 + HALF]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), dpi=120)
    fig.subplots_adjust(top=0.82, bottom=0.05, wspace=0.12)

    axes[0].imshow(scene, cmap="gray", vmin=0, vmax=1)
    if scanning:
        axes[0].add_patch(patches.Rectangle((cur_x0, cur_y0), WIN, WIN, edgecolor=BOX_COLOR,
                                             facecolor="none", linewidth=2.4))
    axes[0].set_title("scene  (current window in yellow)" if scanning
                       else "scene", fontsize=9.5, color=TEXT_COLOR)
    axes[0].axis("off")

    im = axes[1].imshow(display, cmap=cmap, vmin=np.nanmin(score_map), vmax=np.nanmax(score_map))
    axes[1].set_title("window classifier score, by window center\n(gray = not yet scanned / no window fits there)",
                       fontsize=9.5, color=TEXT_COLOR)
    axes[1].axis("off")

    fig.suptitle("sliding the window across the scene", fontsize=13, fontweight="bold", color=TEXT_COLOR)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


X_STEP = 5  # subsample columns within each row so the box visibly sweeps
X_STOPS = x0_range[::X_STEP]
if X_STOPS[-1] != x0_range[-1]:
    X_STOPS.append(x0_range[-1])

frames = []
revealed = set()
for y0 in y0_range:
    for x0 in X_STOPS:
        frames.append(render(revealed, y0, x0, scanning=True))
    revealed.add(y0)

# final frame: full map, no scanning box, held
frames += [render(revealed, None, None, scanning=False)] * 70

frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=50, loop=0)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
