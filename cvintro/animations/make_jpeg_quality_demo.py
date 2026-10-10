"""Generate jpeg_quality_demo.gif: an animated version of the lesson 17
notebook's "A closer look at JPEG artifacts" cell. Sweeps the JPEG quality
factor from very low to very high on the same roofline-against-sky crop of
`house.png`, zoomed 4x, showing blocking and ringing artifacts fade out as
quality increases.

Run from anywhere:
    python make_jpeg_quality_demo.py
Output:
    jpeg_quality_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "jpeg_quality_demo.gif")
IMG_DIR = os.path.join(OUT_DIR, "..", "img")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"

photo = cv2.cvtColor(cv2.imread(os.path.join(IMG_DIR, "house.png")), cv2.COLOR_BGR2RGB)
photo_bgr = cv2.cvtColor(photo, cv2.COLOR_RGB2BGR)  # cv2.imencode expects BGR input

y0, y1, x0, x1 = 10, 60, 50, 150  # roofline against the sky, same crop as the notebook
zoom = 4
original_crop = cv2.resize(photo[y0:y1, x0:x1], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)


def jpeg_crop_at(quality):
    ok, encoded = cv2.imencode(".jpg", photo_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    decoded_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(decoded_rgb[y0:y1, x0:x1], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    return crop, len(encoded)


def render(quality, final_hold=False):
    jpeg_crop, size_bytes = jpeg_crop_at(quality)
    size_kb = size_bytes / 1024

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3), dpi=120)
    fig.subplots_adjust(top=0.80, bottom=0.03, wspace=0.12)
    ax_orig, ax_jpeg = axes

    ax_orig.imshow(original_crop)
    ax_orig.set_title("original crop (zoomed 4x)", fontsize=10.5)
    ax_orig.axis("off")

    ax_jpeg.imshow(jpeg_crop)
    label_color = MATCH_COLOR if final_hold else ACTIVE_COLOR
    # fixed-width (monospace, space-padded) fields so the text doesn't
    # shift horizontally as the digit count changes across the sweep
    ax_jpeg.set_title(f"JPEG quality {quality:3d}\n({size_kb:5.1f} kB)", fontsize=10.5,
                        color=label_color, family="monospace")
    ax_jpeg.axis("off")

    fig.suptitle("JPEG quality sweep", fontsize=12,
                 color=NEUTRAL_COLOR, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


qualities = list(range(5, 101, 5))

frames = []
HOLD_START = 14
HOLD_STEP = 4
HOLD_END = 36

frames += [render(qualities[0])] * HOLD_START
for q in qualities[1:-1]:
    frames += [render(q)] * HOLD_STEP
frames += [render(qualities[-1], final_hold=True)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=90,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
