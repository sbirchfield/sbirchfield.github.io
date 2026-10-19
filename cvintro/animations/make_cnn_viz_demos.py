"""Generate saliency_gradcam_demo.gif and shortcut_detection_demo.gif:
animated companions to the lesson 39 notebook's "Saliency maps" /
"Grad-CAM" / "Do these maps actually track what the model uses?" sections.
Both reuse the exact same two trained models (model_original, trained
clean; model_shortcut, trained with a 5x5 white-corner marker added to
every automobile training image), so this one script produces both GIFs
rather than the course's usual one-script-per-gif pattern -- the same
justified exception make_confusion_matrix_demos.py uses.

  - saliency_gradcam_demo.gif: one image, saliency map fading in next to
    it, then the coarser Grad-CAM heatmap fading in beside that.
  - shortcut_detection_demo.gif: a 2x3 grid -- rows are saliency and
    Grad-CAM (labeled with a shaded tag so the row identity is unmissable),
    columns are the book's three cases (model_original/marked,
    model_shortcut/marked, model_shortcut/clean) -- cycling through
    example images and showing each method's heatmap with a white/black
    crosshair marking the exact peak pixel, plus a compact text readout
    (not a bar chart) of the running and full-test-set "peak lands in the
    marker region" percentages, matching the book's table exactly (same
    computation as peak_in_marker_frac()).

Reuses the notebook's own CNN/saliency_map/grad_cam/add_marker verbatim
(data_rng seed=4, 300 training epochs, lr=0.001).

Run from anywhere:
    python make_cnn_viz_demos.py
Output:
    saliency_gradcam_demo.gif, shortcut_detection_demo.gif
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
OUT_SAL_GIF = os.path.join(OUT_DIR, "saliency_gradcam_demo.gif")
OUT_SHORTCUT_GIF = os.path.join(OUT_DIR, "shortcut_detection_demo.gif")

TEXT_COLOR = "#222222"
HIT_COLOR = "#2e9e44"
MISS_COLOR = "#d62728"
MARKER_COLOR = "#f2c200"

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


def train_cnn(Xt_np, yt_np, seed):
    torch.manual_seed(seed)
    model = CNN()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    Xt = torch.tensor(Xt_np).permute(0, 3, 1, 2); yt = torch.tensor(yt_np)
    for _ in range(300):
        opt.zero_grad()
        out, _ = model(Xt)
        loss = F.binary_cross_entropy_with_logits(out, yt)
        loss.backward()
        opt.step()
    return model


def add_marker(imgs):
    out = imgs.copy()
    out[:, 0:5, 0:5, :] = 1.0
    return out


print("training model_original...")
model_original = train_cnn(X_train, y_train, seed=0)

print("training model_shortcut...")
X_train_shortcut = np.concatenate([X_cat_train, add_marker(X_auto_train)])
model_shortcut = train_cnn(X_train_shortcut, y_train, seed=0)

X_auto_test_marked = add_marker(X_auto_test)


def acc_of(m, X, y):
    with torch.no_grad():
        out, _ = m(torch.tensor(X).permute(0, 3, 1, 2))
        return ((out > 0).float() == torch.tensor(y)).float().mean().item()


X_test_marked = np.concatenate([X_cat_test, X_auto_test_marked])
print(f"model_original: clean={acc_of(model_original, X_test, y_test):.1%} "
      f"marked={acc_of(model_original, X_test_marked, y_test):.1%}")
print(f"model_shortcut: clean={acc_of(model_shortcut, X_test, y_test):.1%} "
      f"marked={acc_of(model_shortcut, X_test_marked, y_test):.1%}")


def saliency_map(model, img_hw3):
    x = torch.tensor(img_hw3).permute(2, 0, 1).unsqueeze(0)
    x.requires_grad_(True)
    score, feat = model(x)
    score.backward()
    return x.grad[0].abs().amax(dim=0).numpy()


def grad_cam(model, img_hw3, out_size=32):
    x = torch.tensor(img_hw3).permute(2, 0, 1).unsqueeze(0)
    x.requires_grad_(True)
    score, feat = model(x)
    feat.retain_grad()
    score.backward()
    weights = feat.grad[0].mean(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * feat[0]).sum(dim=0))
    cam_up = F.interpolate(cam[None, None], size=(out_size, out_size), mode="bilinear", align_corners=False)
    return cam_up[0, 0].detach().numpy()


def peak_rc(heatmap):
    return np.unravel_index(heatmap.argmax(), heatmap.shape)


# ---------------------------------------------------------------------------
# saliency_gradcam_demo.gif
# ---------------------------------------------------------------------------
demo_idx = 103  # an automobile test image (index >= 100)
demo_img = X_test[demo_idx]
demo_saliency = saliency_map(model_original, demo_img)
demo_cam = grad_cam(model_original, demo_img)


def render_sal(step):
    show_maps = step >= 1

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.4), dpi=120)
    fig.subplots_adjust(top=0.88, bottom=0.05, wspace=0.15)

    axes[0].imshow(demo_img)
    axes[0].set_title("input image", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(demo_img)
    if show_maps:
        axes[1].imshow(demo_saliency, cmap="hot", alpha=0.75)
    axes[1].set_title("saliency map" if show_maps else "saliency map (pixel-level)", fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(demo_img)
    if show_maps:
        axes[2].imshow(demo_cam, cmap="hot", alpha=0.75, extent=(0, 32, 32, 0))
    axes[2].set_title("Grad-CAM" if show_maps else "Grad-CAM (coarser, semantic)", fontsize=10)
    axes[2].axis("off")

    fig.suptitle("two ways to ask 'where is the model looking?'", fontsize=12.5,
                  color=TEXT_COLOR, fontweight="bold", y=0.99)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


sal_frames = []
for step, hold in enumerate([70, 130]):
    sal_frames += [render_sal(step)] * hold

sal_frames[0].save(OUT_SAL_GIF, save_all=True, append_images=sal_frames[1:], duration=55, loop=0)
print(f"Saved {OUT_SAL_GIF} ({len(sal_frames)} frames)")

# ---------------------------------------------------------------------------
# shortcut_detection_demo.gif
# ---------------------------------------------------------------------------
GROUPS = [
    ("model_original", model_original, X_auto_test_marked, "marked"),
    ("model_shortcut", model_shortcut, X_auto_test_marked, "marked"),
    ("model_shortcut", model_shortcut, X_auto_test, "clean"),
]
METHODS = [("saliency", saliency_map), ("grad_cam", grad_cam)]
METHOD_LABELS = {"saliency": "saliency", "grad_cam": "Grad-CAM"}
N_SHOWN = 12  # example images animated per group (full-set tallies still used for final %)


def frac_in_marker(model, imgs, heat_fn):
    hits = sum(1 for img in imgs if all(c < 5 for c in peak_rc(heat_fn(model, img))))
    return hits / len(imgs)


group_data = {}
for method_name, heat_fn in METHODS:
    cols = []
    for name, model, imgs, label in GROUPS:
        full_frac = frac_in_marker(model, imgs, heat_fn)
        shown_imgs = imgs[:N_SHOWN]
        heats = [heat_fn(model, img) for img in shown_imgs]
        hits = [1 if all(c < 5 for c in peak_rc(h)) else 0 for h in heats]
        cols.append(dict(name=name, label=label, imgs=shown_imgs, heats=heats, hits=hits, full_frac=full_frac))
        print(f"{name} ({label}) {method_name}: peak-in-marker = {full_frac:.1%} (full test set)")
    group_data[method_name] = cols


# Rows are methods (saliency, Grad-CAM), columns are the three book cases,
# all advancing in sync (same image index everywhere each frame) -- so
# every panel's position never changes, and the six running tallies can be
# compared directly instead of having to remember an earlier panel. Each
# panel's tally is a one-line text readout rather than a bar chart, since
# the exact percentage is the only thing that matters here.
def render_shortcut(img_idx):
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.6), dpi=120)
    fig.subplots_adjust(top=0.84, bottom=0.07, left=0.09, right=0.98, wspace=0.25, hspace=0.45)

    for row, (method_name, _) in enumerate(METHODS):
        for col, g in enumerate(group_data[method_name]):
            ax = axes[row, col]
            img = g["imgs"][img_idx]
            heat = g["heats"][img_idx]
            hit = g["hits"][img_idx]
            running = np.mean(g["hits"][:img_idx + 1])

            ax.imshow(img)
            ax.imshow(heat, cmap="hot", alpha=0.65, extent=(0, 32, 32, 0))
            ax.add_patch(Rectangle((0, 0), 5, 5, fill=False, edgecolor=MARKER_COLOR, linewidth=2.0))
            peak_row, peak_col = peak_rc(heat)
            peak_x, peak_y = peak_col + 0.5, peak_row + 0.5  # center of that heat cell, same extent as the heat overlay
            ax.plot(peak_x, peak_y, marker="x", color="black", markersize=15, mew=4.5, zorder=5)
            ax.plot(peak_x, peak_y, marker="x", color="white", markersize=11, mew=2.2, zorder=6)
            border_color = HIT_COLOR if hit else MISS_COLOR
            if row == 0:
                ax.set_title(f"{g['name']}\n{g['label']} test images", fontsize=9.5,
                              color=TEXT_COLOR, fontweight="bold")
            if col == 0:
                ax.set_ylabel(METHOD_LABELS[method_name], fontsize=13,
                               color=TEXT_COLOR, fontweight="bold", labelpad=10,
                               bbox=dict(boxstyle="round,pad=0.35", facecolor="#eaeaea", edgecolor="none"))
            status = "IN marker" if hit else "elsewhere"
            ax.set_xlabel(f"{status}  ·  running {running:.0%}  ·  full {g['full_frac']:.0%}",
                           fontsize=7.8, color=border_color, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2.2)

    fig.suptitle(f"do saliency / Grad-CAM peaks land in the marker region?  ({img_idx + 1}/{N_SHOWN} images shown)",
                  fontsize=12, color=TEXT_COLOR, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


shortcut_frames = []
HOLD_STEP = 16
HOLD_END = 100
for ii in range(N_SHOWN):
    is_last = (ii == N_SHOWN - 1)
    shortcut_frames += [render_shortcut(ii)] * (HOLD_END if is_last else HOLD_STEP)

shortcut_frames[0].save(OUT_SHORTCUT_GIF, save_all=True, append_images=shortcut_frames[1:],
                          duration=55, loop=0)
print(f"Saved {OUT_SHORTCUT_GIF} ({len(shortcut_frames)} frames)")
