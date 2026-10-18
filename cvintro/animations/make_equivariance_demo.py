"""Generate equivariance_demo.gif: an animated companion to the lesson 34
notebook's "Why translation equivariance actually matters: a generalization
test" section. Trains the notebook's own MLPClassifier and CNNClassifier
(verbatim architectures and train_and_eval() hyperparameters) on shapes
placed only near the center, then slides a single held-out 'plus' shape
from the trained-on center region out to an unseen corner position,
showing each model's live predicted probability as the shape moves --
making the static train/test accuracy numbers (reported in the following
markdown cell) visible as a continuous failure (MLP) vs non-failure (CNN).

Run from anywhere:
    python make_equivariance_demo.py
Output:
    equivariance_demo.gif  (written next to this script)
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
OUT_GIF = os.path.join(OUT_DIR, "equivariance_demo.gif")

MLP_COLOR = "#9467bd"
CNN_COLOR = "#2ca02c"
CORRECT_COLOR = "#2e9e44"
WRONG_COLOR = "#d62728"
TRAIN_ZONE_COLOR = "#dbeedb"
TEST_ZONE_COLOR = "#fde0c4"


def make_image(shape_type, cx, cy, size=16):
    img = np.zeros((size, size), dtype=np.float32)
    if shape_type == "plus":
        img[cy - 1:cy + 2, cx - 3:cx + 4] = 1.0
        img[cy - 3:cy + 4, cx - 1:cx + 2] = 1.0
    else:
        yy, xx = np.mgrid[0:size, 0:size]
        img[((xx - cx) ** 2 + (yy - cy) ** 2) <= 9] = 1.0
    return img


def make_dataset(rng_local, n, position_range):
    imgs, labels = [], []
    for _ in range(n):
        shape_type = rng_local.choice(["plus", "circle"])
        cx, cy = rng_local.integers(*position_range), rng_local.integers(*position_range)
        imgs.append(make_image(shape_type, cx, cy))
        labels.append(0.0 if shape_type == "plus" else 1.0)
    return np.array(imgs, dtype=np.float32), np.array(labels, dtype=np.float32)


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(16 * 16, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 5, padding=2), nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
        )
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1)).squeeze(-1)


def train_model(model_cls, Xtr, ytr, seed, epochs=400, lr=0.01):
    torch.manual_seed(seed)
    model = model_cls()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(Xtr), ytr)
        loss.backward()
        opt.step()
    model.eval()
    return model


# Same dataset and training as the notebook's own generalization-test cells.
data_rng = np.random.default_rng(1)
X_train, y_train = make_dataset(data_rng, 300, (5, 11))
Xtr_t = torch.tensor(X_train).unsqueeze(1)
ytr_t = torch.tensor(y_train)

mlp = train_model(MLPClassifier, Xtr_t, ytr_t, seed=0)
cnn = train_model(CNNClassifier, Xtr_t, ytr_t, seed=0)

with torch.no_grad():
    mlp_train_acc = ((mlp(Xtr_t) > 0).float() == ytr_t).float().mean().item()
    cnn_train_acc = ((cnn(Xtr_t) > 0).float() == ytr_t).float().mean().item()
print(f"train accuracy: mlp={mlp_train_acc:.1%} cnn={cnn_train_acc:.1%}")

# Slide a single 'circle' shape (label 1) through several positions safely
# inside the trained-on sampling range (5, 11), then one step further to
# cx=cy=4, inside the notebook's own disjoint, never-sampled test range
# (3, 5). The MLP stays confidently correct the whole time it's within the
# training range, then flips to confidently *wrong* the moment it crosses
# into the held-out range -- it memorized *where* a feature tends to fire,
# not *what* it looks like. The CNN stays at P(circle) = 1.000 throughout.
POSITIONS = [9, 8, 7, 5, 4]  # 5 and above: within the sampled training range;
                              # 4: inside the notebook's disjoint test range (3, 5),
                              # genuinely never seen during training
TRAIN_RANGE = (5, 11)


def predict(model, img):
    x = torch.tensor(img, dtype=torch.float32).reshape(1, 1, 16, 16)
    with torch.no_grad():
        logit = model(x).item()
    return 1 / (1 + np.exp(-logit))  # P(circle); true label is 'circle' (1)


def render(step):
    cx = cy = POSITIONS[step]
    img = make_image("circle", cx, cy)
    p_mlp = predict(mlp, img)
    p_cnn = predict(cnn, img)
    in_train_zone = TRAIN_RANGE[0] <= cx < TRAIN_RANGE[1]

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(8.8, 4.4), dpi=120,
                                           gridspec_kw={"width_ratios": [1.0, 1.1]})
    fig.subplots_adjust(top=0.78, bottom=0.12, wspace=0.35)

    ax_img.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    zone = "trained here" if in_train_zone else "never seen in training"
    zone_color = TRAIN_ZONE_COLOR if in_train_zone else TEST_ZONE_COLOR
    ax_img.set_title(f"circle at ({cx}, {cy}) -- {zone}", fontsize=10.5,
                       color="#222222", backgroundcolor=zone_color)

    models = ["MLP\n(flatten)", "CNN\n(conv+pool)"]
    probs = [p_mlp, p_cnn]
    bar_colors = [CORRECT_COLOR if p > 0.5 else WRONG_COLOR for p in probs]

    ax_bar.bar(models, probs, color=bar_colors, width=0.55)
    ax_bar.axhline(0.5, color="#888888", linestyle="--", linewidth=1)
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_bar.set_ylabel("P(circle)  [true label: circle]")
    for i, p in enumerate(probs):
        ax_bar.text(i, p + 0.03, f"{p:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax_bar.set_title("above the line = correct", fontsize=10)

    mlp_wrong = p_mlp <= 0.5
    status = "MLP: confidently wrong once outside its training region" if (not in_train_zone and mlp_wrong) \
        else "both still correct" if (p_mlp > 0.5 and p_cnn > 0.5) else "watch the MLP..."
    color = WRONG_COLOR if (not in_train_zone and mlp_wrong) else "#222222"
    fig.suptitle(status, fontsize=12, color=color, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 55
HOLD_END = 90
for step in range(len(POSITIONS)):
    frames += [render(step)] * (HOLD_END if step == len(POSITIONS) - 1 else HOLD_STEP)

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=55,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
