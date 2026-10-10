"""Generate affine_rectify_demo.gif: an animated version of the lesson 8
notebook's "Fitting an affine transform from point correspondences" cell.
Instead of three static panels (original, warped, rectified), this animates
the actual warp happening (canonical -> warped) and then the recovered
transform undoing it (warped -> rectified), with the 3 landmark points
tracked throughout, plus a side panel spelling out that the two stages are
NOT mirror images of each other: stage 1 applies a known matrix directly,
while stage 2 must first *solve* for that matrix from the 3 point
correspondences before it can be applied.

Uses the same test image and warp parameters as the notebook cell.

Run from anywhere:
    python make_affine_rectify_demo.py
Output:
    affine_rectify_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "affine_rectify_demo.gif")

MATCH_COLOR = "#2e9e44"
ACTIVE_COLOR = "#e07b00"
POINT_COLOR = "#e74c3c"
NEUTRAL_COLOR = "#333333"
DIM_COLOR = "#666666"


def make_f_image(size=160):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    color = (255, 200, 0)
    cv2.rectangle(img, (40, 20), (65, 140), color, -1)
    cv2.rectangle(img, (40, 20), (120, 45), color, -1)
    cv2.rectangle(img, (40, 65), (100, 90), color, -1)
    return img


img = make_f_image()
h, w = img.shape[:2]
canvas_size = (w + 60, h + 60)

theta = np.radians(20)
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
shear = np.array([[1, 0.25], [0, 1]])
A_true = R @ shear * 1.1
t_true = np.array([40.0, 20.0])

canonical_pts = np.float32([[40, 20], [120, 20], [40, 140]])
warped_pts = (A_true @ canonical_pts.T).T + t_true

M_recovered = cv2.getAffineTransform(warped_pts.astype(np.float32), canonical_pts.astype(np.float32))
A_rec = M_recovered[:, :2]
t_rec = M_recovered[:, 2]

I2 = np.eye(2)

# Pre-render the fully warped image once -- stage 2 un-warps this fixed image.
M_true_full = np.hstack([A_true, t_true.reshape(2, 1)])
warped_img = cv2.warpAffine(img, M_true_full, canvas_size)


def apply_2x2(pts, A, t):
    return pts @ A.T + t


def stage1_frame(alpha):
    A = I2 + alpha * (A_true - I2)
    t = alpha * t_true
    M = np.hstack([A, t.reshape(2, 1)])
    canvas = cv2.warpAffine(img, M, canvas_size)
    pts = apply_2x2(canonical_pts, A, t)
    return canvas, pts, A, t


def stage2_frame(alpha):
    A = I2 + alpha * (A_rec - I2)
    t = alpha * t_rec
    M = np.hstack([A, t.reshape(2, 1)])
    canvas = cv2.warpAffine(warped_img, M, canvas_size)
    pts = apply_2x2(warped_pts, A, t)
    return canvas, pts, A, t


def fmt_matrix(A, t):
    return (f"A = [[{A[0, 0]:5.2f}, {A[0, 1]:5.2f}],\n"
            f"     [{A[1, 0]:5.2f}, {A[1, 1]:5.2f}]]\n"
            f"t = [{t[0]:5.1f}, {t[1]:5.1f}]")


def render(canvas, pts, side_lines):
    fig = plt.figure(figsize=(9.2, 6.0), dpi=120)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.62], left=0.02, right=0.98,
                            top=0.97, bottom=0.03, wspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax_side = fig.add_subplot(gs[1])

    ax.imshow(canvas)
    ax.scatter(pts[:, 0], pts[:, 1], c=POINT_COLOR, s=60, zorder=3, edgecolors="white", linewidths=1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax_side.axis("off")
    ax_side.set_xlim(0, 1)
    ax_side.set_ylim(0, 1)
    y = 0.95
    for text, fontsize, color, weight, family in side_lines:
        ax_side.text(0.0, y, text, fontsize=fontsize, color=color, fontweight=weight,
                      family=family, ha="left", va="top")
        y -= 0.08 * (text.count("\n") + 1) * (fontsize / 12.0) + 0.045

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def side_lines_stage1(A, t):
    return [
        ("Stage 1", 17, ACTIVE_COLOR, "bold", "sans-serif"),
        ("apply the warp", 14.5, ACTIVE_COLOR, "bold", "sans-serif"),
        ("p' = Ap + t", 14.5, NEUTRAL_COLOR, "normal", "sans-serif"),
        (fmt_matrix(A, t), 14, ACTIVE_COLOR, "normal", "monospace"),
        ("(normally unknown)", 12.5, DIM_COLOR, "normal", "sans-serif"),
    ]


def side_lines_stage2(A, t, revealed):
    lines = [
        ("Stage 2", 17, ACTIVE_COLOR, "bold", "sans-serif"),
        ("solve for the warp", 14.5, ACTIVE_COLOR, "bold", "sans-serif"),
        ("3 points → solve for A, t", 14.5, NEUTRAL_COLOR, "normal", "sans-serif"),
    ]
    if revealed:
        lines.append((fmt_matrix(A_rec, t_rec), 14, MATCH_COLOR, "normal", "monospace"))
        lines.append(("apply A, t to every pixel", 12.5, DIM_COLOR, "normal", "sans-serif"))
    return lines


N = 24
HOLD_START = 12
HOLD_MID = 16
HOLD_END = 44

alphas = np.linspace(0.0, 1.0, N + 1)

frames = []

canvas0, pts0, A0, t0 = stage1_frame(0.0)
intro_lines = [
    ("original photo", 17, NEUTRAL_COLOR, "bold", "sans-serif"),
    ("3 known landmarks", 14.5, NEUTRAL_COLOR, "normal", "sans-serif"),
]
frames += [render(canvas0, pts0, intro_lines)] * HOLD_START

for a in alphas[1:]:
    canvas, pts, A, t = stage1_frame(a)
    frames.append(render(canvas, pts, side_lines_stage1(A, t)))

canvas_w, pts_w, A_w, t_w = stage1_frame(1.0)
mid_lines = side_lines_stage1(A_w, t_w) + [
    ("now treat as unknown", 12.5, DIM_COLOR, "normal", "sans-serif"),
]
frames += [render(canvas_w, pts_w, mid_lines)] * HOLD_MID

for a in alphas[1:]:
    canvas, pts, A, t = stage2_frame(a)
    frames.append(render(canvas, pts, side_lines_stage2(A, t, revealed=True)))

canvas_r, pts_r, A_r, t_r = stage2_frame(1.0)
final_lines = side_lines_stage2(A_r, t_r, revealed=True) + [
    ("matches the true warp ✓", 13.5, MATCH_COLOR, "bold", "sans-serif"),
]
frames += [render(canvas_r, pts_r, final_lines)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=80,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
