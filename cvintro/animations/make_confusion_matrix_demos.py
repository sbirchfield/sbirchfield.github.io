"""Generate confusion_matrix_demo.gif, tpfpfn_demo.gif, and
precision_recall_demo.gif: animated companions to the lesson 36 notebook's
"Confusion matrix", "True positives, false positives, true negatives,
false negatives", and "Precision and recall" sections. All three reuse the
exact same trained 4-class CNN and confusion matrix (so they share one
training run instead of three), which is why this one script produces all
three GIFs rather than the course's usual one-script-per-gif pattern.

  - confusion_matrix_demo.gif: the 4x4 confusion matrix filling in one
    cell at a time (row by row), diagonal cells colored as hits, off-
    diagonal cells colored as misses, ending on the same heatmap as the
    notebook's static imshow.
  - tpfpfn_demo.gif: cycles through each of the 4 classes, collapsing the
    full matrix down to that class's TP/FP/FN/TN 2x2 table -- the same
    collapse the notebook's own "truck" example performs, generalized to
    every class.
  - precision_recall_demo.gif: cycles through each class again, this time
    showing precision and recall as literal fractions of the same TP/FP/FN
    counts -- a recall bar split into TP (caught) + FN (missed), a
    precision bar split into TP (correct) + FP (false alarm). The notebook
    computes precision/recall directly from hard argmax counts with no
    adjustable decision threshold anywhere, so this -- not a threshold
    sweep -- is the animation actually grounded in what the lesson does.

Reuses the notebook's own CIFAR loading, CNN architecture, and training
hyperparameters (seed=0, lr=0.0007, 150 epochs) verbatim, so the confusion
matrix matches the static cells exactly.

Run from anywhere:
    python make_confusion_matrix_demos.py
Output:
    confusion_matrix_demo.gif, tpfpfn_demo.gif, precision_recall_demo.gif
    (written next to this script)
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
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CM_GIF = os.path.join(OUT_DIR, "confusion_matrix_demo.gif")
OUT_TP_GIF = os.path.join(OUT_DIR, "tpfpfn_demo.gif")

HIT_COLOR = "#2e9e44"
MISS_COLOR = "#d62728"
TP_COLOR = "#2e9e44"
FN_COLOR = "#e07b00"
FP_COLOR = "#9467bd"
TN_COLOR = "#dddddd"
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


CM_CACHE = OUT_DIR + "/.cm_cache.npy"
if os.path.exists(CM_CACHE):
    cached = np.load(CM_CACHE, allow_pickle=True).item()
    cm, acc = cached["cm"], cached["acc"]
    print(f"(loaded cached confusion matrix; delete {CM_CACHE} to retrain)")
else:
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
        preds = model(Xte).argmax(1).numpy()

    acc = (preds == y_test).mean()
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_test, preds):
        cm[t, p] += 1
    np.save(CM_CACHE, {"cm": cm, "acc": acc})

print(f"overall test accuracy: {acc:.1%}")
print("confusion matrix:\n", cm)

# ---------------------------------------------------------------------------
# confusion_matrix_demo.gif: fill in the matrix one cell at a time
# ---------------------------------------------------------------------------
cells_order = [(i, j) for i in range(4) for j in range(4)]


def render_cm(step):
    fig, ax = plt.subplots(figsize=(5.2, 4.8), dpi=120)
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES, rotation=30, fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES, fontsize=9)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(3.5, -0.5)
    ax.set_aspect("equal")

    is_stepping = step < len(cells_order)
    cur = cells_order[min(step, len(cells_order) - 1)]

    for idx, (i, j) in enumerate(cells_order):
        revealed = idx < step or (idx == step and is_stepping) or not is_stepping
        if not revealed:
            face, text, tcolor = "#ffffff", "", None
        else:
            val = cm[i, j]
            is_hit = i == j
            if idx == step and is_stepping:
                face = "#eafaea" if is_hit else "#fbe3e3"
                tcolor = HIT_COLOR if is_hit else MISS_COLOR
            else:
                intensity = min(val / cm.max(), 1.0)
                face = plt.cm.Blues(0.15 + 0.7 * intensity)
                tcolor = "white" if intensity > 0.5 else "#222222"
            text = f"{val}"
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face,
                                 edgecolor="#999999", linewidth=0.8, zorder=0))
        if text:
            ax.text(j, i, text, ha="center", va="center", fontsize=12,
                     color=tcolor, fontweight="bold", zorder=1)
        if idx == step and is_stepping:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                     edgecolor=HIT_COLOR if i == j else MISS_COLOR,
                                     linewidth=2.6, zorder=2))

    if is_stepping:
        i, j = cur
        kind = "a hit (diagonal)" if i == j else "a miss"
        title = f"true={CLASSES[i]}, predicted={CLASSES[j]}: {cm[i,j]} examples -- {kind}"
        color = HIT_COLOR if i == j else MISS_COLOR
    else:
        title = f"overall test accuracy: {acc:.1%}  (rows=true, cols=predicted)"
        color = NEUTRAL_COLOR
    fig.suptitle(title, fontsize=11, color=color, fontweight="bold")
    fig.subplots_adjust(top=0.84, bottom=0.20)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 30
HOLD_END = 90
total = len(cells_order)
for step in range(total + 1):
    frames += [render_cm(step)] * (HOLD_END if step == total else HOLD_STEP)

frames[0].save(OUT_CM_GIF, save_all=True, append_images=frames[1:], duration=55, loop=0)
print(f"Saved {OUT_CM_GIF} ({len(frames)} frames)")

# ---------------------------------------------------------------------------
# tpfpfn_demo.gif: cycle through each class, collapsing to its TP/FP/FN/TN
# ---------------------------------------------------------------------------


def render_tp(class_idx):
    cls_name = CLASSES[class_idx]
    tp = cm[class_idx, class_idx]
    fn = cm[class_idx, :].sum() - tp
    fp = cm[:, class_idx].sum() - tp
    tn = cm.sum() - tp - fn - fp

    fig, (ax_cm, ax_2x2) = plt.subplots(1, 2, figsize=(9.2, 4.6), dpi=120,
                                          gridspec_kw={"width_ratios": [1.1, 1.0]})
    fig.subplots_adjust(top=0.78, bottom=0.20, wspace=0.4)

    ax_cm.set_xticks(range(4)); ax_cm.set_xticklabels(CLASSES, rotation=30, fontsize=8)
    ax_cm.set_yticks(range(4)); ax_cm.set_yticklabels(CLASSES, fontsize=8)
    ax_cm.set_xlabel("predicted", fontsize=9); ax_cm.set_ylabel("true", fontsize=9)
    ax_cm.set_xlim(-0.5, 3.5); ax_cm.set_ylim(3.5, -0.5)
    ax_cm.set_aspect("equal")
    for i in range(4):
        for j in range(4):
            if i == class_idx and j == class_idx:
                face, tcolor = "#eafaea", TP_COLOR
            elif i == class_idx:
                face, tcolor = "#fde9d1", FN_COLOR
            elif j == class_idx:
                face, tcolor = "#ede3f5", FP_COLOR
            else:
                face, tcolor = TN_COLOR, "#666666"
            ax_cm.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face,
                                        edgecolor="#999999", linewidth=0.8, zorder=0))
            ax_cm.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9,
                        color=tcolor, fontweight="bold", zorder=1)
    ax_cm.set_title(f"'{cls_name}' vs. everything else", fontsize=10)

    ax_2x2.axis("off")
    table_data = [
        ["", "pred.\nYES", "pred.\nNO"],
        ["actual\nYES", f"TP = {tp}", f"FN = {fn}"],
        ["actual\nNO", f"FP = {fp}", f"TN = {tn}"],
    ]
    colors = [
        ["white", "white", "white"],
        ["white", "#eafaea", "#fde9d1"],
        ["white", "#ede3f5", TN_COLOR],
    ]
    tbl = ax_2x2.table(cellText=table_data, cellColours=colors, loc="center",
                         cellLoc="center", colWidths=[0.3, 0.35, 0.35])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.6)
    ax_2x2.text(0.5, 0.08, f"\"YES\" = {cls_name}", ha="center", fontsize=9,
                 color=NEUTRAL_COLOR, style="italic", transform=ax_2x2.transAxes)

    fig.suptitle(f"class: {cls_name}", fontsize=13, fontweight="bold",
                 color=NEUTRAL_COLOR)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames2 = []
HOLD = 70
for ci in range(4):
    frames2 += [render_tp(ci)] * HOLD

frames2[0].save(OUT_TP_GIF, save_all=True, append_images=frames2[1:], duration=55, loop=0)
print(f"Saved {OUT_TP_GIF} ({len(frames2)} frames)")

# ---------------------------------------------------------------------------
# precision_recall_demo.gif: cycle through each class, showing precision
# and recall as literal fractions of the same TP/FP/FN counts
# ---------------------------------------------------------------------------
OUT_PR_GIF = os.path.join(OUT_DIR, "precision_recall_demo.gif")


def render_pr(class_idx):
    cls_name = CLASSES[class_idx]
    tp = cm[class_idx, class_idx]
    fn = cm[class_idx, :].sum() - tp
    fp = cm[:, class_idx].sum() - tp
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    fig, (ax_r, ax_p) = plt.subplots(1, 2, figsize=(8.4, 4.6), dpi=120)
    fig.subplots_adjust(top=0.76, bottom=0.14, wspace=0.5)

    def stacked_bar(ax, hit, miss, hit_label, miss_label, hit_color, miss_color, frac, frac_label):
        total = hit + miss
        ax.bar([0], [hit], color=hit_color, width=0.5, label=hit_label)
        ax.bar([0], [miss], bottom=[hit], color=miss_color, width=0.5, label=miss_label)
        ax.set_xlim(-0.6, 1.3)
        ax.set_ylim(0, total * 1.12 if total > 0 else 1)
        ax.set_xticks([])
        ax.text(0, total * 1.04, f"n = {total}", ha="center", fontsize=9)
        ax.text(0.65, total / 2, f"{frac_label}\n= {frac:.2f}", ha="left", va="center",
                 fontsize=12, fontweight="bold", color=TP_COLOR)
        ax.legend(fontsize=8, loc="lower right")

    stacked_bar(ax_r, tp, fn, f"TP = {tp}  (caught)", f"FN = {fn}  (missed)",
                TP_COLOR, FN_COLOR, recall, "recall\nTP/(TP+FN)")
    ax_r.set_title(f"actually {cls_name}", fontsize=10.5)

    stacked_bar(ax_p, tp, fp, f"TP = {tp}  (correct)", f"FP = {fp}  (false alarm)",
                TP_COLOR, FP_COLOR, precision, "precision\nTP/(TP+FP)")
    ax_p.set_title(f"predicted {cls_name}", fontsize=10.5)

    fig.suptitle(f"class: {cls_name}", fontsize=13, fontweight="bold", color=NEUTRAL_COLOR)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames3 = []
for ci in range(4):
    frames3 += [render_pr(ci)] * HOLD

frames3[0].save(OUT_PR_GIF, save_all=True, append_images=frames3[1:], duration=55, loop=0)
print(f"Saved {OUT_PR_GIF} ({len(frames3)} frames)")
