"""Generate pr_sweep_demo.gif: an animated companion to the lesson 36
notebook's "The precision-recall curve" section. Shows the *mechanism*
behind the already-static precision-recall curve: a decision threshold
sweeping across each test image's softmax "truck" score, with the
resulting TP/FP/FN/TN classification shown directly on a sorted-score
strip, and the (recall, precision) point crawling along the same curve
the notebook's static figure shows.

Tightly scoped to this one mechanism -- it does not re-derive precision/
recall from scratch (that's precision_recall_demo.gif's job) or re-walk
the confusion matrix (confusion_matrix_demo.gif / tpfpfn_demo.gif). It
retrains its own copy of the identical CNN (same architecture, seed=0,
lr=0.0007, 150 epochs, same CIFAR subset) to get per-example softmax
scores, which the other three animations' shared training run doesn't
cache.

Run from anywhere:
    python make_pr_sweep_demo.py
Output:
    pr_sweep_demo.gif  (written next to this script)
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
OUT_GIF = os.path.join(OUT_DIR, "pr_sweep_demo.gif")

TP_COLOR = "#2e9e44"
FN_COLOR = "#e07b00"
FP_COLOR = "#9467bd"
TN_COLOR = "#dddddd"
CURVE_COLOR = "#1f6fb4"
THRESH_COLOR = "#d62728"
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
CLASSES = ["cat", "dog", "automobile", "truck"]
CLASS_IDS = {name: CIFAR_LABELS.index(name) for name in CLASSES}

train_imgs, train_labels = [], []
for i in range(1, 6):
    imgs, labels = load_cifar_batch(CACHE_DIR / f"data_batch_{i}")
    train_imgs.append(imgs); train_labels.append(labels)
train_imgs, train_labels = np.concatenate(train_imgs), np.concatenate(train_labels)
test_imgs, test_labels = load_cifar_batch(CACHE_DIR / "test_batch")

rng = np.random.default_rng(7)


def subset_for_classes(imgs, labels, per_class_counts):
    out_imgs, out_labels = [], []
    for i, name in enumerate(CLASSES):
        idx = np.where(labels == CLASS_IDS[name])[0]
        idx = rng.permutation(idx)[:per_class_counts[name]]
        out_imgs.append(imgs[idx])
        out_labels.append(np.full(len(idx), i, dtype=np.int64))
    out_imgs, out_labels = np.concatenate(out_imgs), np.concatenate(out_labels)
    perm = rng.permutation(len(out_imgs))
    return out_imgs[perm], out_labels[perm]


X_train, y_train = subset_for_classes(train_imgs, train_labels,
    {"cat": 400, "dog": 400, "automobile": 400, "truck": 90})
X_test, y_test = subset_for_classes(test_imgs, test_labels,
    {"cat": 150, "dog": 150, "automobile": 150, "truck": 150})


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
        )
        self.fc = nn.Linear(64, 4)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


torch.manual_seed(0)
model = CNN()
opt = torch.optim.Adam(model.parameters(), lr=0.0007)
Xt = torch.tensor(X_train).permute(0, 3, 1, 2)
yt = torch.tensor(y_train)
for _ in range(150):
    opt.zero_grad()
    loss = F.cross_entropy(model(Xt), yt)
    loss.backward()
    opt.step()

Xte = torch.tensor(X_test).permute(0, 3, 1, 2)
with torch.no_grad():
    logits = model(Xte)
    preds = logits.argmax(1).numpy()
    probs = F.softmax(logits, dim=1).numpy()

acc = (preds == y_test).mean()
print(f"overall test accuracy: {acc:.1%}")

cls = CLASSES.index("truck")
scores = probs[:, cls]
is_truck = (y_test == cls)

order = np.argsort(-scores)  # descending, for the left-panel strip
sorted_scores = scores[order]
sorted_is_truck = is_truck[order]
n = len(scores)

# full curve, for the right panel background
curve_thresholds = np.linspace(0, 1, 101)
curve_precisions, curve_recalls = [], []
for t in curve_thresholds:
    pred_pos = scores >= t
    tp_t = (pred_pos & is_truck).sum()
    fp_t = (pred_pos & ~is_truck).sum()
    fn_t = (~pred_pos & is_truck).sum()
    curve_precisions.append(tp_t / (tp_t + fp_t) if tp_t + fp_t > 0 else 1.0)
    curve_recalls.append(tp_t / (tp_t + fn_t) if tp_t + fn_t > 0 else 0.0)

# monotonic envelope (same construction as the notebook's static figure): at
# each recall, take the best precision achieved at that recall or higher.
# Ties in recall are collapsed to their best precision first -- otherwise a
# later, worse-precision point at the same recall can undercut an earlier,
# better one.
_order = np.argsort(curve_recalls)
_r_sorted = np.array(curve_recalls)[_order]
_p_sorted = np.array(curve_precisions)[_order]
curve_r_sorted, _inverse = np.unique(_r_sorted, return_inverse=True)
_p_uniq = np.zeros_like(curve_r_sorted)
np.maximum.at(_p_uniq, _inverse, _p_sorted)
curve_p_envelope = np.maximum.accumulate(_p_uniq[::-1])[::-1]

_brute = np.array([max(p for p, r in zip(curve_precisions, curve_recalls) if r >= rv)
                    for rv in curve_r_sorted])
assert np.allclose(curve_p_envelope, _brute), "envelope does not match brute-force definition"
assert np.all(np.diff(curve_p_envelope) <= 1e-12), "envelope is not monotonically non-increasing"
print("envelope verified against brute-force max-at-or-above-recall definition")

# argmax operating point, for reference (same as the static figure)
tp0 = int(((preds == cls) & is_truck).sum())
fn0 = int(((preds != cls) & is_truck).sum())
fp0 = int(((preds == cls) & ~is_truck).sum())
argmax_recall = tp0 / (tp0 + fn0)
argmax_precision = tp0 / (tp0 + fp0)

SWEEP_THRESHOLDS = np.concatenate([[1.02], np.linspace(1.0, 0.0, 41), [-0.02]])


def render(step):
    t = SWEEP_THRESHOLDS[step]
    pred_pos = sorted_scores >= t
    outcome = np.where(pred_pos & sorted_is_truck, 0,
              np.where(pred_pos & ~sorted_is_truck, 1,
              np.where(~pred_pos & sorted_is_truck, 2, 3)))
    colors = np.array([TP_COLOR, FP_COLOR, FN_COLOR, TN_COLOR])[outcome]

    tp = int((pred_pos & sorted_is_truck).sum())
    fp = int((pred_pos & ~sorted_is_truck).sum())
    fn = int((~pred_pos & sorted_is_truck).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 1.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    fig, (ax_strip, ax_curve) = plt.subplots(1, 2, figsize=(10.6, 4.8), dpi=120,
                                               gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.subplots_adjust(top=0.80, bottom=0.13, wspace=0.32, left=0.07, right=0.97)

    ax_strip.scatter(np.arange(n), sorted_scores, c=colors, s=14, edgecolor="none", zorder=2)
    ax_strip.axhline(t, color=THRESH_COLOR, linewidth=1.8, linestyle="--", zorder=3)
    if t > 0.88:
        text_y, va = t - 0.07, "top"
    else:
        text_y, va = t + 0.07, "bottom"
    ax_strip.text(n * 0.02, text_y, f"threshold = {t:.2f}", color=THRESH_COLOR,
                   ha="left", va=va, fontsize=9.5, fontweight="bold", zorder=4,
                   bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax_strip.set_xlim(-n * 0.02, n * 1.02)
    ax_strip.set_ylim(-0.05, 1.05)
    ax_strip.set_xlabel("test images, sorted by score")
    ax_strip.set_ylabel("softmax score for 'truck'")
    ax_strip.set_title("each dot = one test image", fontsize=10.5)
    for color, label in [(TP_COLOR, "TP"), (FP_COLOR, "FP"), (FN_COLOR, "FN"), (TN_COLOR, "TN")]:
        ax_strip.scatter([], [], c=color, label=label, s=30)
    ax_strip.legend(fontsize=8, loc="center right", framealpha=0.9)

    is_final = (step == len(SWEEP_THRESHOLDS) - 1)
    if is_final:
        # only impose the monotonic envelope once the sweep is done -- during
        # the sweep itself, only the raw (noisy) curve is shown.
        ax_curve.plot(curve_recalls, curve_precisions, color="#aac4e0", linewidth=1.2,
                       zorder=1, label="raw (noisy)")
        ax_curve.step(curve_r_sorted, curve_p_envelope, where="pre", color=CURVE_COLOR,
                       linewidth=2.2, zorder=2, label="envelope (monotonic)")
    else:
        ax_curve.plot(curve_recalls, curve_precisions, color=CURVE_COLOR, linewidth=1.8, zorder=1)
    ax_curve.scatter([argmax_recall], [argmax_precision], color=THRESH_COLOR, marker="x",
                       s=50, zorder=3, label="hard-argmax point")
    ax_curve.scatter([recall], [precision], color=THRESH_COLOR, s=90, zorder=4,
                       edgecolor="white", linewidth=1.2, label="current threshold")
    ax_curve.set_xlim(-0.03, 1.03)
    ax_curve.set_ylim(0.0, 1.05)
    ax_curve.set_xlabel("recall")
    ax_curve.set_ylabel("precision")
    ax_curve.set_title("precision-recall curve", fontsize=10.5)
    ax_curve.legend(fontsize=8, loc="lower left")

    fig.suptitle(f"TP={tp}  FP={fp}  FN={fn}    precision={precision:.2f}  recall={recall:.2f}",
                  fontsize=12, color=TEXT_COLOR, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 6
HOLD_END = 90
n_steps = len(SWEEP_THRESHOLDS)
for step in range(n_steps):
    frames += [render(step)] * (HOLD_END if step in (0, n_steps - 1) else HOLD_STEP)

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=55,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
