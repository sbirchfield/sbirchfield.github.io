"""Generate document_rectification_demo.gif: an animated version of the
lesson 24 notebook's "Perspective correction: rectifying a photographed
document" cell. Uses the exact same document image and 4 corner
correspondences as the notebook. Two stages: first the 4 corners move from
the document's own rectangle to the keystoned "photographed" quadrilateral,
warping the document live (same effect as applying H_distort); then the
corners move back from the keystoned quadrilateral to the rectangle,
un-warping it (same effect as applying H_rectify). Each frame's homography
is solved fresh from the 4 (partially interpolated) point correspondences
via cv2.getPerspectiveTransform, so every frame is a genuine homography
between the fixed source corners and the current 4 target points -- not a
blended matrix.

Run from anywhere:
    python make_document_rectification_demo.py
Output:
    document_rectification_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "document_rectification_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#333333"
POINT_COLOR = "#e74c3c"

# Same document image and corner correspondences as the notebook cell.
document = np.full((300, 220, 3), 255, dtype=np.uint8)
cv2.rectangle(document, (20, 20), (200, 280), (0, 0, 0), 3)
cv2.putText(document, 'HELLO', (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

h, w = document.shape[:2]
corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
photographed_corners = np.float32([[45, 15], [175, 25], [210, 290], [5, 280]])

H_distort = cv2.getPerspectiveTransform(corners, photographed_corners)
photographed = cv2.warpPerspective(document, H_distort, (w, h))


def warp_with_corners(img, src_fixed, dst_corners, size):
    H = cv2.getPerspectiveTransform(src_fixed, dst_corners.astype(np.float32))
    return cv2.warpPerspective(img, H, size)


def render(canvas_rgb, pts, title, title_color):
    fig = plt.figure(figsize=(5.4, 6.0), dpi=120)
    ax = fig.add_axes([0.02, 0.08, 0.96, 0.86])
    ax.imshow(canvas_rgb)
    ax.scatter(pts[:, 0], pts[:, 1], c=POINT_COLOR, s=70, zorder=3,
               edgecolors="white", linewidths=1.3)
    ax.set_xlim(-30, w + 30)
    ax.set_ylim(h + 30, -30)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle(title, fontsize=13, color=title_color, fontweight="bold", y=0.985)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


N = 24
HOLD_START = 14
HOLD_MID = 20
HOLD_END = 44

alphas = np.linspace(0.0, 1.0, N + 1)
frames = []

# Stage 1: rectangle -> keystoned photograph (document distorted live).
canvas0 = cv2.cvtColor(document, cv2.COLOR_BGR2RGB)
frames += [render(canvas0, corners, "original document (4 known corners)", NEUTRAL_COLOR)] * HOLD_START

for a in alphas[1:]:
    pts = (1 - a) * corners + a * photographed_corners
    warped = warp_with_corners(document, corners, pts, (w, h))
    frames.append(render(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), pts,
                           "stage 1: photographing at an angle", ACTIVE_COLOR))

canvas_mid = cv2.cvtColor(photographed, cv2.COLOR_BGR2RGB)
frames += [render(canvas_mid, photographed_corners,
                    "keystoned photo -- now treat corners as known clicks", ACTIVE_COLOR)] * HOLD_MID

# Stage 2: keystoned quadrilateral -> rectangle (un-warping back to upright).
for a in alphas[1:]:
    pts = (1 - a) * photographed_corners + a * corners
    rectified_partial = warp_with_corners(photographed, photographed_corners, pts, (w, h))
    frames.append(render(cv2.cvtColor(rectified_partial, cv2.COLOR_BGR2RGB), pts,
                           "stage 2: rectifying from the 4 corner clicks", MATCH_COLOR))

canvas_end = warp_with_corners(photographed, photographed_corners, corners, (w, h))
frames += [render(cv2.cvtColor(canvas_end, cv2.COLOR_BGR2RGB), corners,
                    "rectified ✓", MATCH_COLOR)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=80,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
