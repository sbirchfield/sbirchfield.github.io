"""Generate regularizer_race_demo.gif: an animated companion to the lesson
35 notebook's three individual fixes (augmentation, weight decay, dropout),
each of which was only ever compared against the unregularized baseline
one at a time via a single final accuracy number. This animation races all
four validation-*accuracy* curves together, live, on the same 12-image toy
dataset: the unregularized baseline stuck low, the three fixes each
climbing higher.

(Validation *loss* was tried first instead of accuracy, but it turned out
to be extremely noisy and seed-dependent on this tiny 12-image dataset over
many epochs -- e.g. dropout's validation loss can balloon well past the
unregularized baseline's even while its accuracy is fine -- which isn't the
effect this lesson's own comparisons are about. Accuracy is the metric the
notebook's own 8-seed-averaged fix comparisons actually report, and is far
more stable here.)

Reuses the notebook's own make_image/make_dataset (data_rng seed=2),
augment() (aug_rng seed=5, 20 copies/image), CNN and CNNDropout
architectures, and hyperparameters (lr=0.01, weight_decay=0.05 for the
weight-decay config, dropout_p=0.5 for the dropout config) verbatim. All
four configs are trained from the same seed=0 for a fair, synchronized
race; this single-seed comparison illustrates the *shape* of the effect
each fix in this lesson's actual 8-seed-averaged numbers.

Run from anywhere:
    python make_regularizer_race_demo.py
Output:
    regularizer_race_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "regularizer_race_demo.gif")

BASELINE_COLOR = "#d62728"
AUG_COLOR = "#1f6fb4"
WD_COLOR = "#e07b00"
DROPOUT_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"


def make_image(shape_type, cx, cy, size=16, rng=None):
    img = np.zeros((size, size), dtype=np.float32)
    if shape_type == "plus":
        img[cy - 1:cy + 2, cx - 3:cx + 4] = 1.0
        img[cy - 3:cy + 4, cx - 1:cx + 2] = 1.0
    else:
        yy, xx = np.mgrid[0:size, 0:size]
        img[((xx - cx) ** 2 + (yy - cy) ** 2) <= 9] = 1.0
    if rng is not None:
        img = np.clip(img + rng.normal(0, 0.4, img.shape), 0, 1).astype(np.float32)
    return img


def make_dataset(rng_local, n, position_range):
    imgs, labels = [], []
    for _ in range(n):
        shape_type = rng_local.choice(["plus", "circle"])
        cx, cy = rng_local.integers(*position_range), rng_local.integers(*position_range)
        imgs.append(make_image(shape_type, cx, cy, rng=rng_local))
        labels.append(0.0 if shape_type == "plus" else 1.0)
    return np.array(imgs, dtype=np.float32), np.array(labels, dtype=np.float32)


def augment(img, rng_local):
    if rng_local.random() < 0.5:
        img = np.fliplr(img).copy()
    if rng_local.random() < 0.5:
        img = np.flipud(img).copy()
    angle = rng_local.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((8, 8), angle, 1.0)
    img = cv2.warpAffine(img, M, (16, 16))
    return img.astype(np.float32)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1)).squeeze(-1)


class CNNDropout(nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
        )
        self.fc1 = nn.Linear(32, 64)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        feat = torch.relu(self.fc1(self.conv(x).flatten(1)))
        return self.fc2(self.dropout(feat)).squeeze(-1)


def train_tracked(model_cls, Xtr, ytr, Xval, yval, epochs, lr, weight_decay=0.0, seed=0):
    """Tracks validation *accuracy*, not loss -- this tiny 12-image dataset
    makes validation BCE loss extremely noisy and seed-dependent over many
    epochs (it can balloon well past the unregularized baseline's loss even
    when accuracy is fine), which isn't the effect this lesson's own
    comparisons are about. Accuracy is the metric the notebook's own
    8-seed-averaged fix comparisons actually report.

    model_cls is a zero-arg callable (not an already-built model) so the
    seed can be set *before* construction, matching the notebook's own
    documented pattern -- otherwise weight init depends on whatever global
    RNG state happens to be left over from training the previous config."""
    torch.manual_seed(seed)
    model = model_cls()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    val_accs = []
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(Xtr), ytr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            acc = ((model(Xval) > 0).float() == yval).float().mean().item()
        val_accs.append(acc)
    return val_accs


# Same dataset as the notebook's "A deliberately hard, small dataset" cell.
data_rng = np.random.default_rng(2)
X_train, y_train = make_dataset(data_rng, 12, (3, 13))
X_val, y_val = make_dataset(data_rng, 150, (3, 13))
Xtr_t = torch.tensor(X_train).unsqueeze(1); ytr_t = torch.tensor(y_train)
Xval_t = torch.tensor(X_val).unsqueeze(1); yval_t = torch.tensor(y_val)

# Same augmented dataset as the notebook's "Fix 1" cell.
aug_rng = np.random.default_rng(5)
X_aug, y_aug = [], []
for _ in range(20):
    for img, label in zip(X_train, y_train):
        X_aug.append(augment(img, aug_rng))
        y_aug.append(label)
X_aug, y_aug = np.array(X_aug, dtype=np.float32), np.array(y_aug, dtype=np.float32)
Xaug_t = torch.tensor(X_aug).unsqueeze(1); yaug_t = torch.tensor(y_aug)

EPOCHS = 200
torch.manual_seed(0)  # re-seeded individually inside train_tracked per config anyway
configs = [
    ("no fix", BASELINE_COLOR, lambda: train_tracked(CNN, Xtr_t, ytr_t, Xval_t, yval_t, EPOCHS, lr=0.01)),
    ("augmentation", AUG_COLOR, lambda: train_tracked(CNN, Xaug_t, yaug_t, Xval_t, yval_t, EPOCHS, lr=0.01)),
    ("weight decay", WD_COLOR, lambda: train_tracked(CNN, Xtr_t, ytr_t, Xval_t, yval_t, EPOCHS, lr=0.01, weight_decay=0.05)),
    ("dropout", DROPOUT_COLOR, lambda: train_tracked(lambda: CNNDropout(0.5), Xtr_t, ytr_t, Xval_t, yval_t, EPOCHS, lr=0.01)),
]

results = {}
for name, color, fn in configs:
    results[name] = fn()
    print(f"{name:>13}: final val acc = {results[name][-1]:.1%}, "
          f"max val acc = {max(results[name]):.1%}")


def render(epoch):
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=120)
    xs = list(range(epoch + 1))
    for name, color, _ in configs:
        vals = results[name][:epoch + 1]
        ax.plot(xs, vals, color=color, linewidth=1.6, label=name)
        ax.scatter([epoch], [vals[-1]], color=color, s=30, zorder=5)

    ax.set_xlim(0, EPOCHS - 1)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"epoch {epoch:3d}", fontsize=10)
    fig.suptitle("validation accuracy, four ways", fontsize=12.5, fontweight="bold",
                 color=NEUTRAL_COLOR)
    fig.subplots_adjust(top=0.86, bottom=0.12)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
CHECKPOINT_EVERY = 3
HOLD_STEP = 2
HOLD_END = 80

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
