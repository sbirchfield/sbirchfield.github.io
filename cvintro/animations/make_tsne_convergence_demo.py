"""Generate tsne_convergence_demo.gif: an animated companion to the lesson
39 notebook's "Feature-space visualization: t-SNE" section. Instead of
showing only the finished embedding, this animates the actual gradient
descent: points start at their random initial positions and are revealed
converging, iteration by iteration, into the two class clusters.

Reuses the notebook's own CNN/tsne()/pooled_features exactly (model_original
trained for 300 epochs on the same cat-vs-automobile split, data_rng
seed=4, tsne seed=0), except tsne() is modified to also return a snapshot
of Y at regular intervals, so the final frame matches the notebook's own
static figure.

Run from anywhere:
    python make_tsne_convergence_demo.py
Output:
    tsne_convergence_demo.gif  (written next to this script)
"""
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
OUT_GIF = os.path.join(OUT_DIR, "tsne_convergence_demo.gif")

CAT_COLOR = "#1f6fb4"
AUTO_COLOR = "#e07b00"
TEXT_COLOR = "#222222"

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
    return imgs[idx].copy()


rng = np.random.default_rng(4)
X_cat_train = take(train_imgs, train_labels, "cat", 300, rng)
X_auto_train = take(train_imgs, train_labels, "automobile", 300, rng)
X_cat_test = take(test_imgs, test_labels, "cat", 100, rng)
X_auto_test = take(test_imgs, test_labels, "automobile", 100, rng)

X_train = np.concatenate([X_cat_train, X_auto_train])
y_train = np.array([0.0] * 300 + [1.0] * 300, dtype=np.float32)
X_test = np.concatenate([X_cat_test, X_auto_test])
y_test = np.array([0.0] * 100 + [1.0] * 100, dtype=np.float32)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2)
        self.gpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        f1 = F.relu(self.conv1(x))
        f2 = F.relu(self.conv2(self.pool(f1)))
        feat = self.gpool(f2).flatten(1)
        return self.fc(feat).squeeze(-1), f2


print("training model_original (300 epochs)...")
torch.manual_seed(0)
model_original = CNN()
opt = torch.optim.Adam(model_original.parameters(), lr=0.001)
Xt = torch.tensor(X_train).permute(0, 3, 1, 2); yt = torch.tensor(y_train)
for _ in range(300):
    opt.zero_grad()
    out, _ = model_original(Xt)
    loss = F.binary_cross_entropy_with_logits(out, yt)
    loss.backward()
    opt.step()

with torch.no_grad():
    out, _ = model_original(torch.tensor(X_test).permute(0, 3, 1, 2))
    acc = ((out > 0).float() == torch.tensor(y_test)).float().mean().item()
print(f"test accuracy: {acc:.1%}")


def pooled_features(m, X):
    with torch.no_grad():
        _, f2 = m(torch.tensor(X).permute(0, 3, 1, 2))
        return F.adaptive_max_pool2d(f2, 1).flatten(1).numpy()


feat_trained = pooled_features(model_original, X_test)


def tsne_tracked(X, n_iter=500, perplexity=15.0, lr=100.0, seed=0, snapshot_every=5):
    """Same as the notebook's tsne(), but also records Y every
    `snapshot_every` iterations so the optimization can be animated."""
    rng_local = np.random.default_rng(seed)
    n = X.shape[0]
    sq_dists = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)

    target_entropy = np.log(perplexity)
    P = np.zeros((n, n))
    for i in range(n):
        others = np.arange(n) != i
        d_i = sq_dists[i, others]
        lo, hi = 1e-4, 1e4
        for _ in range(50):
            beta = (lo + hi) / 2
            p = np.exp(-d_i * beta)
            p /= p.sum() + 1e-12
            entropy = -np.sum(p * np.log(p + 1e-12))
            if entropy > target_entropy:
                lo = beta
            else:
                hi = beta
        P[i, others] = p
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12) * 4.0

    Y = rng_local.normal(0, 1e-2, (n, 2))
    velocity = np.zeros_like(Y)
    snapshots = [Y.copy()]
    for it in range(n_iter):
        if it == 100:
            P /= 4.0
        d = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
        num = 1.0 / (1.0 + d)
        np.fill_diagonal(num, 0)
        Q = np.maximum(num / num.sum(), 1e-12)
        coeff = (P - Q) * num
        grad = 4 * (coeff[:, :, None] * (Y[:, None, :] - Y[None, :, :])).sum(1)
        momentum = 0.5 if it < 100 else 0.8
        velocity = momentum * velocity - lr * grad
        Y = Y + velocity
        if (it + 1) % snapshot_every == 0:
            snapshots.append(Y.copy())
    return snapshots


print("running t-SNE with snapshots...")
snapshots = tsne_tracked(feat_trained, seed=0)
print(f"{len(snapshots)} snapshots")

# sanity check against the book's own reported metric: a 1-nearest-neighbor
# check in the final 2D embedding should land around 78% (trained features)
_Yf = snapshots[-1]
_labels = np.array([0] * 100 + [1] * 100)
_d = ((_Yf[:, None, :] - _Yf[None, :, :]) ** 2).sum(-1)
np.fill_diagonal(_d, np.inf)
_nn = _d.argmin(axis=1)
_nn_acc = (_labels[_nn] == _labels).mean()
print(f"1-NN accuracy in final embedding: {_nn_acc:.1%} (book reports ~78%)")


def render(step):
    Y = snapshots[step]
    # fix axis limits to the final embedding's range so points don't appear
    # to jitter the whole canvas as they move -- the convergence itself is
    # the point, not an ever-rescaling viewport
    final = snapshots[-1]
    pad = 0.15 * (final.max() - final.min())
    lo, hi = final.min() - pad, final.max() + pad

    fig, ax = plt.subplots(figsize=(6.0, 5.4), dpi=120)
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.1, right=0.97)

    ax.scatter(*Y[:100].T, c=CAT_COLOR, s=18, label="cat", zorder=2)
    ax.scatter(*Y[100:].T, c=AUTO_COLOR, s=18, label="automobile", zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=9, loc="upper right")

    is_final = (step == len(snapshots) - 1)
    it = min(step * 5, 500)
    if is_final:
        title = "converged (iteration 500)"
    else:
        title = f"t-SNE embedding of trained CNN features -- iteration {it}"
    fig.suptitle(title, fontsize=12, color=TEXT_COLOR, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 2
HOLD_END = 100
n_steps = len(snapshots)
for step in range(n_steps):
    frames += [render(step)] * (HOLD_END if step == n_steps - 1 else HOLD_STEP)

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=55,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
