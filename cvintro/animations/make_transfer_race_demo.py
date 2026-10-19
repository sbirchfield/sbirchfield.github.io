"""Generate transfer_race_demo.gif: an animated companion to the lesson 38
notebook's "Three strategies" section. Races target-task validation
accuracy, live over training, for the three strategies the notebook
compares only via a single final-accuracy table: from scratch, frozen
backbone, and fine-tuned.

Reuses the notebook's own Backbone/Classifier/train_source/train_target
setup (same data_rng seed=3 source/target split, same train_source(seed=0)
pretraining run reused across all target strategies, same hyperparameters:
200 epochs, lr=0.01, backbone_lr=0.0003 for fine-tuning) verbatim, except
train_target is extended to record test accuracy every epoch instead of
only the final value. Averaged over the same 8 seeds as the notebook's own
table, so the final frame's numbers match it.

Run from anywhere:
    python make_transfer_race_demo.py
Output:
    transfer_race_demo.gif  (written next to this script)
"""
import copy
import io
import os
import pickle
import tarfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "transfer_race_demo.gif")

SCRATCH_COLOR = "#d62728"
FROZEN_COLOR = "#1f6fb4"
FINETUNE_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CACHE_ROOT = Path.home() / ".cache" / "cvintro"
CACHE_DIR = CACHE_ROOT / "cifar-10-batches-py"


def ensure_cifar10():
    if CACHE_DIR.exists():
        return
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    archive_path = CACHE_ROOT / "cifar-10-python.tar.gz"
    if not archive_path.exists():
        print("Downloading CIFAR-10 (~163 MB, one-time, cached under ~/.cache/cvintro)...")
        urllib.request.urlretrieve(CIFAR_URL, archive_path)
    print("Extracting...")
    with tarfile.open(archive_path) as tar:
        tar.extractall(CACHE_ROOT)


def load_cifar_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    imgs = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
    labels = np.array(d[b"labels"], dtype=np.int64)
    return imgs, labels


ensure_cifar10()

CIFAR_LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CID = {name: CIFAR_LABELS.index(name) for name in CIFAR_LABELS}

train_imgs, train_labels = [], []
for i in range(1, 6):
    imgs, labels = load_cifar_batch(CACHE_DIR / f"data_batch_{i}")
    train_imgs.append(imgs); train_labels.append(labels)
train_imgs, train_labels = np.concatenate(train_imgs), np.concatenate(train_labels)
test_imgs, test_labels = load_cifar_batch(CACHE_DIR / "test_batch")


def take(imgs, labels, name, n, rng_local):
    idx = np.where(labels == CID[name])[0]
    idx = rng_local.permutation(idx)[:n]
    return imgs[idx]


data_rng = np.random.default_rng(3)

# source task: airplane vs. automobile, plenty of data
src_a = take(train_imgs, train_labels, "airplane", 300, data_rng)
src_b = take(train_imgs, train_labels, "automobile", 300, data_rng)
X_src = np.concatenate([src_a, src_b])
y_src = np.concatenate([np.zeros(300), np.ones(300)]).astype(np.int64)
perm = data_rng.permutation(len(X_src)); X_src, y_src = X_src[perm], y_src[perm]

# target task: cat vs. truck, neither ever seen by the source task, deliberately starved
X_tgt_train = np.concatenate([take(train_imgs, train_labels, "cat", 15, data_rng),
                               take(train_imgs, train_labels, "truck", 15, data_rng)])
y_tgt_train = np.array([0] * 15 + [1] * 15, dtype=np.int64)
X_tgt_test = np.concatenate([take(test_imgs, test_labels, "cat", 150, data_rng),
                              take(test_imgs, test_labels, "truck", 150, data_rng)])
y_tgt_test = np.array([0] * 150 + [1] * 150, dtype=np.int64)

Xtr = torch.tensor(X_tgt_train).permute(0, 3, 1, 2); ytr = torch.tensor(y_tgt_train)
Xte = torch.tensor(X_tgt_test).permute(0, 3, 1, 2); yte = torch.tensor(y_tgt_test)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
        )

    def forward(self, x):
        return self.conv(x).flatten(1)


class Classifier(nn.Module):
    def __init__(self, backbone, freeze_backbone):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        feat = self.backbone(x)
        if self.freeze_backbone:
            feat = feat.detach()
        return self.fc(feat)


def train_source(seed, epochs=300, lr=0.001):
    torch.manual_seed(seed)
    backbone = Backbone()
    fc = nn.Linear(32, 2)
    opt = torch.optim.Adam(list(backbone.parameters()) + list(fc.parameters()), lr=lr)
    Xt = torch.tensor(X_src).permute(0, 3, 1, 2); yt = torch.tensor(y_src)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(fc(backbone(Xt)), yt)
        loss.backward()
        opt.step()
    return backbone


def train_target_tracked(backbone, freeze_backbone, seed, epochs=200, lr=0.01, backbone_lr=None):
    """Same as the notebook's train_target, but records test accuracy every
    epoch instead of only returning the final value."""
    torch.manual_seed(seed)
    model = Classifier(backbone, freeze_backbone)
    if freeze_backbone:
        opt = torch.optim.Adam(model.fc.parameters(), lr=lr)
    elif backbone_lr is not None:
        opt = torch.optim.Adam([
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": model.fc.parameters(), "lr": lr},
        ])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    accs = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(Xtr), ytr)
        loss.backward()
        opt.step()
        with torch.no_grad():
            accs.append((model(Xte).argmax(1) == yte).float().mean().item())
    return accs


EPOCHS = 200
N_SEEDS = 8
pretrained = train_source(seed=0)

configs = [
    ("from scratch", SCRATCH_COLOR,
     lambda seed: train_target_tracked(Backbone(), freeze_backbone=False, seed=seed, epochs=EPOCHS)),
    ("frozen backbone", FROZEN_COLOR,
     lambda seed: train_target_tracked(copy.deepcopy(pretrained), freeze_backbone=True, seed=seed, epochs=EPOCHS)),
    ("fine-tuned", FINETUNE_COLOR,
     lambda seed: train_target_tracked(copy.deepcopy(pretrained), freeze_backbone=False, seed=seed,
                                         epochs=EPOCHS, backbone_lr=0.0003)),
]

results = {}
for name, color, fn in configs:
    curves = np.array([fn(seed) for seed in range(N_SEEDS)])  # (N_SEEDS, EPOCHS)
    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0)
    results[name] = (mean_curve, std_curve)
    print(f"{name:>16}: final mean acc = {mean_curve[-1]:.1%} "
          f"(std across seeds = {curves[:, -1].std():.1%})")


def render(epoch):
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=120)
    xs = list(range(epoch + 1))
    for name, color, _ in configs:
        mean_curve, std_curve = results[name]
        vals = mean_curve[:epoch + 1]
        stds = std_curve[:epoch + 1]
        ax.fill_between(xs, vals - stds, vals + stds, color=color, alpha=0.15, linewidth=0, zorder=1)
        ax.plot(xs, vals, color=color, linewidth=1.6, label=name, zorder=3)
        ax.scatter([epoch], [vals[-1]], color=color, s=30, zorder=5)

    ax.set_xlim(0, EPOCHS - 1)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("target test accuracy (mean ± 1σ over 8 seeds)")
    # legend order matches the plot's vertical stacking at the final epoch
    # (fine-tuned highest/green, frozen backbone middle/blue, from scratch
    # lowest/red), not the training/plotting order above.
    handles, labels = ax.get_legend_handles_labels()
    legend_order = ["fine-tuned", "frozen backbone", "from scratch"]
    by_label = dict(zip(labels, handles))
    ax.legend([by_label[name] for name in legend_order], legend_order, fontsize=8, loc="lower right")
    ax.set_title(f"epoch {epoch:3d}", fontsize=10)
    fig.suptitle("transfer learning: three strategies", fontsize=12.5,
                  fontweight="bold", color=NEUTRAL_COLOR)
    fig.subplots_adjust(top=0.86, bottom=0.12)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
CHECKPOINT_EVERY = 3
HOLD_STEP = 2
HOLD_END = 90

checkpoints = list(range(0, EPOCHS, CHECKPOINT_EVERY))
if checkpoints[-1] != EPOCHS - 1:
    checkpoints.append(EPOCHS - 1)

for epoch in checkpoints:
    frames += [render(epoch)] * HOLD_STEP
frames += [frames[-1]] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=55,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
