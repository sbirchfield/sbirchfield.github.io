"""Generate structure_tensor_sweep_demo.gif: an animated companion to the
lesson 20 notebook's "structure tensor" cell. Slides the same 5x5 window used
there across the notebook's own corners/edges/flat test image, sweeping from
a flat region through a straight edge to a true corner. At each window
position, shows the local gradient scatter (Ix, Iy) for every pixel in the
window and its structure-tensor eigen-ellipse, live-classifying the point as
flat / edge / corner from the two eigenvalues -- exactly the rule stated in
the notebook's own markdown ("both small: flat", "one large one small: edge",
"both large: corner").

Run from anywhere:
    python make_structure_tensor_sweep_demo.py
Output:
    structure_tensor_sweep_demo.gif  (written next to this script)
"""
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "structure_tensor_sweep_demo.gif")

ACTIVE_COLOR = "#e07b00"   # edge
MATCH_COLOR = "#2e9e44"    # corner
NEUTRAL_COLOR = "#222222"
FLAT_COLOR = "#888888"     # flat

# Same test image as the notebook's structure-tensor cell, but with a little
# noise added so the (Ix, Iy) scatter in a window is actually a *cloud* of
# points instead of collapsing onto the handful of exact gradient values a
# noise-free piecewise-constant image produces (which would be invisible as
# a single dot in flat regions and a couple of overlapping dots on edges).
img_clean = np.zeros((100, 100), dtype=np.uint8)
cv2.rectangle(img_clean, (20, 20), (80, 80), 200, -1)
cv2.rectangle(img_clean, (40, 40), (60, 60), 100, -1)

rng = np.random.default_rng(0)
img_f = img_clean.astype(np.float64) + rng.normal(0, 6.0, img_clean.shape)
img = np.clip(img_f, 0, 255).astype(np.uint8)

Ix = cv2.Sobel(img_f, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(img_f, cv2.CV_64F, 0, 1, ksize=3)

R = 5  # same window radius as the notebook's `window = 5` (here as a half-width)


def structure_tensor(cx, cy):
    y0, y1 = cy - R, cy + R + 1
    x0, x1 = cx - R, cx + R + 1
    ix = Ix[y0:y1, x0:x1].ravel()
    iy = Iy[y0:y1, x0:x1].ravel()
    # Mean-centered covariance of the (Ix, Iy) samples in the window, rather
    # than the raw (uncentered) sum of outer products: a hard edge pushes
    # most of a window's gradient samples to one side of the origin, so an
    # origin-centered ellipse from the raw second moment sits mostly off of
    # the visible point cloud. The covariance ellipse is centered on the
    # cloud's own centroid, so what's drawn is an honest fit to the dots
    # actually on screen; the flat/edge/corner intuition (one or both axes
    # of spread being large) is unaffected by the centering choice.
    pts = np.stack([ix, iy], axis=1)
    mean = pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]  # descending: lambda1 >= lambda2
    return eigvals[order], eigvecs[:, order], ix, iy, mean


def classify(eigvals):
    lam1, lam2 = eigvals
    if lam1 < 2000:
        return "flat", FLAT_COLOR
    elif lam2 < 2000:
        return "edge", ACTIVE_COLOR
    else:
        return "corner", MATCH_COLOR


THETA = np.linspace(0, 2 * np.pi, 100)
CIRCLE = np.stack([np.cos(THETA), np.sin(THETA)])


ELLIPSE_SIGMA = 2.5  # roughly bounds the bulk of each window's gradient cloud


def render(cx, cy):
    eigvals, eigvecs, ix, iy, mean = structure_tensor(cx, cy)
    label, color = classify(eigvals)
    radii = ELLIPSE_SIGMA * np.sqrt(np.clip(eigvals, 0, None))

    fig, (ax_img, ax_grad) = plt.subplots(
        1, 2, figsize=(9.2, 4.6), dpi=120,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.05, right=0.97, wspace=0.22)

    ax_img.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax_img.add_patch(Rectangle((cx - R - 0.5, cy - R - 0.5), 2 * R + 1, 2 * R + 1,
                                 fill=False, edgecolor=color, linewidth=2.0))
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title("5x5 window sliding across the image", fontsize=10.5)

    GRAD_LIM = 460  # wide enough that the strongest edge/corner windows don't clip
    ax_grad.set_xlim(-GRAD_LIM, GRAD_LIM)
    ax_grad.set_ylim(GRAD_LIM, -GRAD_LIM)
    ax_grad.set_aspect("equal")
    ax_grad.axhline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax_grad.axvline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax_grad.scatter(ix, iy, s=14, color="#4c72b0", alpha=0.6, zorder=2)

    ellipse = mean[:, None] + (eigvecs * radii) @ CIRCLE
    ax_grad.plot(ellipse[0], ellipse[1], color=color, linewidth=2.2, zorder=3)
    for j in range(2):
        tip = mean + eigvecs[:, j] * radii[j]
        ax_grad.annotate("", xy=(tip[0], tip[1]), xytext=(mean[0], mean[1]),
                          arrowprops=dict(arrowstyle="->", color=color, lw=1.8), zorder=4)
    ax_grad.set_xticks([])
    ax_grad.set_yticks([])
    ax_grad.set_title("gradients (Ix, Iy) in the window\n+ structure-tensor eigen-ellipse",
                        fontsize=10.5)

    fig.suptitle(f"classified as: {label}", fontsize=13, color=color, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# A closed square loop -- down onto the inner square's top edge, left along
# that edge into its corner, up off the corner back through the edge, right
# back to the start -- so the window just "moves around the image" and loops
# seamlessly, rather than tracing one specific path that needs describing.
# Each leg is axis-aligned so the window only ever crosses the ONE boundary
# relevant to that leg's story (a diagonal path would clip the corner early
# and confuse the flat->edge leg).
WAYPOINTS = [(50, 30), (50, 40), (40, 40), (40, 30), (50, 30)]
HOLD_WAYPOINT = 16
STEPS_BETWEEN = 18

frames = [render(*WAYPOINTS[0])] * HOLD_WAYPOINT
for (x0, y0), (x1, y1) in zip(WAYPOINTS[:-1], WAYPOINTS[1:]):
    for t in np.linspace(0, 1, STEPS_BETWEEN)[1:]:
        cx = int(round(x0 + t * (x1 - x0)))
        cy = int(round(y0 + t * (y1 - y0)))
        frames.append(render(cx, cy))
    frames += [render(x1, y1)] * HOLD_WAYPOINT

frames += [frames[-1]] * 20

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=70,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
