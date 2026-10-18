"""Generate augmentation_gallery_demo.gif: an animated companion to the
lesson 35 notebook's "Fix 1: data augmentation" section. Takes a single
training image and cycles it through several different random variants
from the notebook's own augment() function (random horizontal/vertical
flip, random rotation up to +/-10 degrees). Shows the full menu of
possible augmentations on one line (gray = not used this frame, blue =
used), with a second line spelling out details (like the rotation angle)
for whichever ones fired -- the same label the whole time, a
different-looking image every step.

Reuses the notebook's own make_image()/make_dataset() (data_rng seed=2) to
reconstruct the exact same 12-image training set, then augment() (same
logic, instrumented to report which transforms fired) on its first image.

Run from anywhere:
    python make_augmentation_gallery_demo.py
Output:
    augmentation_gallery_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "augmentation_gallery_demo.gif")

TEXT_COLOR = "#222222"
LABEL_COLOR = "#1f6fb4"
OFF_COLOR = "#aaaaaa"
SEP_COLOR = "#cccccc"


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


def augment_labeled(img, rng_local):
    """Same logic as the notebook's augment(), but also reports which
    transforms fired, for the animation's caption."""
    results = []  # (name, applied, detail or None)

    did_flip_h = rng_local.random() < 0.5
    if did_flip_h:
        img = np.fliplr(img).copy()
    results.append(("flip-h", did_flip_h, None))

    did_flip_v = rng_local.random() < 0.5
    if did_flip_v:
        img = np.flipud(img).copy()
    results.append(("flip-v", did_flip_v, None))

    angle = rng_local.uniform(-10, 10)
    rotate_applied = abs(angle) >= 1.0
    M = cv2.getRotationMatrix2D((8, 8), angle, 1.0)
    img = cv2.warpAffine(img, M, (16, 16))
    results.append(("rotate", rotate_applied, f"{angle:+.1f}°" if rotate_applied else None))

    return img.astype(np.float32), results


# Same dataset as the notebook's "A deliberately hard, small dataset" cell.
data_rng = np.random.default_rng(2)
X_train, y_train = make_dataset(data_rng, 12, (3, 13))
base_img = X_train[0]
base_label = "plus" if y_train[0] == 0.0 else "circle"

# Same augmentation rng as the notebook's own augment() cell.
aug_rng = np.random.default_rng(5)
N_VARIANTS = 6
variants = [(base_img, None)]  # frame 0: the original, unaugmented
for _ in range(N_VARIANTS):
    variants.append(augment_labeled(base_img, aug_rng))

ALL_NAMES = ["flip-h", "flip-v", "rotate"]


def rainbow_text(fig, y, tokens, fontsize=10.5, fontweight="bold"):
    """tokens: list of (text, color). Draws them left-to-right, centered
    as a whole, on one horizontal line at figure-fraction height y."""
    renderer = fig.canvas.get_renderer()
    texts = []
    for s, c in tokens:
        t = fig.text(0, y, s, color=c, fontsize=fontsize, fontweight=fontweight,
                      ha="left", va="center")
        texts.append(t)
    widths = [t.get_window_extent(renderer=renderer).width for t in texts]
    total_width = sum(widths)
    fig_width_px = fig.get_size_inches()[0] * fig.dpi
    x = 0.5 - (total_width / fig_width_px) / 2
    for t, w in zip(texts, widths):
        t.set_position((x, y))
        x += w / fig_width_px


def render(step):
    img, results = variants[step]

    fig = plt.figure(figsize=(4.4, 5.0), dpi=120)
    ax = fig.add_axes([0.05, 0.02, 0.90, 0.76])
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])

    fig.text(0.5, 0.975, f"label: {base_label}  (unchanged)", fontsize=10.5, color=TEXT_COLOR,
               ha="center", va="center")

    if results is None:
        fig.text(0.5, 0.90, "original training image", fontsize=11, color=TEXT_COLOR,
                   fontweight="bold", ha="center", va="center")
    else:
        applied = {name: (ok, detail) for name, ok, detail in results}
        tokens = []
        for i, name in enumerate(ALL_NAMES):
            if i > 0:
                tokens.append((" · ", SEP_COLOR))
            ok, _ = applied[name]
            color = LABEL_COLOR if ok else OFF_COLOR
            tokens.append((name, color))
        fig.canvas.draw()
        rainbow_text(fig, 0.90, tokens, fontsize=11, fontweight="bold")

        detail_parts = [f"{name}: {detail}" for name, (ok, detail) in applied.items() if ok and detail]
        if detail_parts:
            fig.text(0.5, 0.845, "  |  ".join(detail_parts), fontsize=9.5, color=LABEL_COLOR,
                       ha="center", va="center")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 55
for step in range(len(variants)):
    frames += [render(step)] * HOLD_STEP

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=55,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
