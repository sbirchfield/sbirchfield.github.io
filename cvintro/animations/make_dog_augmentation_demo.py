"""Generate dog_augmentation_demo.gif: a companion to the lesson 35
notebook's "Fix 1: data augmentation" section, right after
augmentation_gallery_demo.gif. That earlier animation applies augment() to
one of the synthetic plus/circle training images actually used in this
lesson's quantitative experiment; this one applies a wider variety of
augmentations (flip, rotation, brightness/contrast/color jitter, Gaussian
noise, blur, occlusion) to a real photo (img/dog_looking.jpg) purely to
show what augmentation can look like on a real image -- it isn't part of
any training run or accuracy number in this lesson.

Each frame shows the full menu of possible augmentations on one line (gray
= not used this frame, blue = used), with a second line spelling out the
specific parameters (angle, brightness delta, etc.) for whichever ones
fired.

Run from anywhere:
    python make_dog_augmentation_demo.py
Output:
    dog_augmentation_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "dog_augmentation_demo.gif")
IMG_PATH = os.path.join(OUT_DIR, "..", "img", "dog_looking.jpg")

TEXT_COLOR = "#222222"
LABEL_COLOR = "#1f6fb4"
OFF_COLOR = "#aaaaaa"
SEP_COLOR = "#cccccc"

base_img = np.asarray(Image.open(IMG_PATH).convert("RGB"), dtype=np.float32) / 255.0
H, W = base_img.shape[:2]


def augment_photo(img, rng_local):
    # unlike the synthetic plus/circle augment(), no vertical flip --
    # an upside-down dog isn't a realistic training example.
    out = img.copy()
    results = []  # (name, applied, detail or None)

    flip = rng_local.random() < 0.5
    if flip:
        out = np.fliplr(out).copy()
    results.append(("flip-h", flip, None))

    angle = rng_local.uniform(-10, 10)
    rotate_applied = abs(angle) >= 1.0
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    out = cv2.warpAffine(out, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    results.append(("rotate", rotate_applied, f"{angle:+.1f}°" if rotate_applied else None))

    do_bright = rng_local.random() < 0.5
    brightness = 0.0
    if do_bright:
        brightness = rng_local.uniform(-0.18, 0.18)
        out = np.clip(out + brightness, 0, 1)
    results.append(("brightness", do_bright, f"{brightness:+.2f}" if do_bright else None))

    do_contrast = rng_local.random() < 0.5
    contrast = 1.0
    if do_contrast:
        contrast = rng_local.uniform(0.7, 1.35)
        out = np.clip((out - 0.5) * contrast + 0.5, 0, 1)
    results.append(("contrast", do_contrast, f"×{contrast:.2f}" if do_contrast else None))

    do_color = rng_local.random() < 0.4
    if do_color:
        tint = rng_local.uniform(0.85, 1.15, size=3).astype(np.float32)
        out = np.clip(out * tint[None, None, :], 0, 1)
    results.append(("color jitter", do_color, None))

    do_noise = rng_local.random() < 0.45
    if do_noise:
        sigma = rng_local.uniform(0.03, 0.09)
        out = np.clip(out + rng_local.normal(0, sigma, out.shape), 0, 1).astype(np.float32)
    results.append(("gaussian noise", do_noise, None))

    do_blur = rng_local.random() < 0.35
    ksize = 0
    if do_blur:
        ksize = int(rng_local.choice([3, 5, 7]))
        out = cv2.GaussianBlur(out, (ksize, ksize), 0)
    results.append(("blur", do_blur, f"k={ksize}" if do_blur else None))

    do_occ = rng_local.random() < 0.35
    if do_occ:
        bw = int(rng_local.integers(W // 6, W // 3))
        bh = int(rng_local.integers(H // 6, H // 3))
        bx = int(rng_local.integers(0, W - bw))
        by = int(rng_local.integers(0, H - bh))
        out = out.copy()
        out[by:by + bh, bx:bx + bw] = rng_local.uniform(0.25, 0.55)
    results.append(("occlusion", do_occ, None))

    return out.astype(np.float32), results


ALL_NAMES = ["flip-h", "rotate", "brightness", "contrast", "color jitter",
             "gaussian noise", "blur", "occlusion"]

aug_rng = np.random.default_rng(7)
N_VARIANTS = 8
variants = [(base_img, None)]  # frame 0: the original, unaugmented
for _ in range(N_VARIANTS):
    variants.append(augment_photo(base_img, aug_rng))


def rainbow_text(fig, y, tokens, fontsize=9.5, fontweight="bold"):
    """tokens: list of (text, color). Draws them left-to-right, centered
    as a whole, on one horizontal line at figure-fraction height y."""
    renderer = fig.canvas.get_renderer()
    texts = []
    for s, c in tokens:
        t = fig.text(0, y, s, color=c, fontsize=fontsize, fontweight=fontweight,
                      ha="left", va="center")
        texts.append(t)
    widths = []
    for t in texts:
        bbox = t.get_window_extent(renderer=renderer)
        widths.append(bbox.width)
    total_width = sum(widths)
    fig_width_px = fig.get_size_inches()[0] * fig.dpi
    x = 0.5 - (total_width / fig_width_px) / 2
    for t, w in zip(texts, widths):
        t.set_position((x, y))
        x += w / fig_width_px


ROW1, ROW2 = ALL_NAMES[:4], ALL_NAMES[4:]


def render(step):
    img, results = variants[step]

    fig = plt.figure(figsize=(4.6, 5.3), dpi=120)
    ax = fig.add_axes([0.03, 0.02, 0.94, 0.77])
    ax.imshow(np.clip(img, 0, 1))
    ax.set_xticks([]); ax.set_yticks([])

    fig.text(0.5, 0.975, "same idea, on a real photo", fontsize=10, color=TEXT_COLOR,
               ha="center", va="center")

    if results is None:
        fig.text(0.5, 0.91, "original photo", fontsize=10.5, color=TEXT_COLOR,
                   fontweight="bold", ha="center", va="center")
    else:
        applied = {name: (ok, detail) for name, ok, detail in results}
        fig.canvas.draw()
        for row, y in ((ROW1, 0.925), (ROW2, 0.885)):
            tokens = []
            for i, name in enumerate(row):
                if i > 0:
                    tokens.append((" · ", SEP_COLOR))
                ok, _ = applied[name]
                color = LABEL_COLOR if ok else OFF_COLOR
                tokens.append((name, color))
            rainbow_text(fig, y, tokens, fontsize=9.5, fontweight="bold")

        detail_parts = [f"{name}: {detail}" for name, (ok, detail) in applied.items() if ok and detail]
        if detail_parts:
            fig.text(0.5, 0.838, "  |  ".join(detail_parts), fontsize=9, color=LABEL_COLOR,
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
