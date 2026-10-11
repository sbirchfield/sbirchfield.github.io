"""Generate epipolar_depth_demo.gif: an animated companion to the lesson 28
notebook's "Why a line? A geometric picture" cell. A 3D point X slides along
camera 1's ray through a fixed pixel x1 -- all depths are equally consistent
with camera 1 alone -- while the epipolar plane spanned by C1, C2, and the
ray stays fixed. Both cameras' image planes are drawn in the same 3D scene
(instead of a separate flat panel) so the ray-to-epipolar-line relationship
is visible directly: x1 sits fixed at the center of image plane 1 (since the
ray IS camera 1's sightline through that pixel), while X's projection onto
image plane 2 slides along the epipolar line -- the plane's edge-on view of
the fixed epipolar plane -- as X moves.

Reuses the notebook cell's own C1, C2, ray_dir geometry so the animation is
visually consistent with the static 3D figure it accompanies; camera 2 gets
a simple pinhole model (own R/forward/right/up, not the notebook's unrelated
synthetic point cloud) purely to place its image plane and render x2. All
geometry is computed in the notebook's own (X, Y, Z) coordinates, then
swapped to (X, Z, Y) only at the point of plotting -- purely a rendering
choice that puts the ray's dominant component on the screen-horizontal axis
instead of the vertical one, so the image planes (always perpendicular to
their camera's viewing direction) render as upright "windows" rather than
near-flat discs, matching the classic epipolar-geometry diagram look.

Run from anywhere:
    python make_epipolar_depth_demo.py
Output:
    epipolar_depth_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "epipolar_depth_demo.gif")

ACTIVE_COLOR = "#e07b00"
MATCH_COLOR = "#2e9e44"
NEUTRAL_COLOR = "#222222"
PLANE1_COLOR = "tab:blue"
PLANE2_COLOR = "tab:green"

# Same geometry as the notebook's "Why a line?" cell.
C1 = np.array([0.0, 0.0, 0.0])
C2 = np.array([3.0, 0.5, -0.5])
ray_dir = np.array([0.6, 0.8, 3.0])
ray_dir = ray_dir / np.linalg.norm(ray_dir)

DEPTH_MIN, DEPTH_MAX = 1.2, 6.0
FAR_POINT = C1 + DEPTH_MAX * ray_dir  # spans the (fixed) epipolar plane triangle


def V(p):
    """Swap Y and Z for plotting only -- see module docstring."""
    return np.array([p[0], p[2], p[1]])


def camera_basis(forward, up_hint=np.array([0.0, -1.0, 0.0])):
    """right, up, forward -- a simple orthonormal camera basis in world space,
    used only to compute WHERE a point lands on the real image plane (its
    local right/up offsets), not to draw the plane itself (see sheared_quad)."""
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up_hint)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return right, up, forward


# World-axis directions used to DRAW each image plane: a simple schematic
# parallelogram, not the (barely legible) true perspective rectangle. Left
# and right edges are pure vertical lines (vary only in world Y); "shear"
# shifts the whole left edge up (positive) or down (negative) relative to
# the right edge, approximating the classic epipolar-geometry diagram look.
X_DIR = np.array([1.0, 0.0, 0.0])
Y_DIR = np.array([0.0, 1.0, 0.0])


def sheared_quad(center, half_w, half_h, shear):
    top_left = center - half_w * X_DIR + (half_h + shear / 2) * Y_DIR
    bottom_left = center - half_w * X_DIR + (-half_h + shear / 2) * Y_DIR
    top_right = center + half_w * X_DIR + (half_h - shear / 2) * Y_DIR
    bottom_right = center + half_w * X_DIR + (-half_h - shear / 2) * Y_DIR
    return [bottom_left, bottom_right, top_right, top_left]


# Camera 1's image plane: its optical axis IS the ray through x1, so x1 sits
# fixed at the plane's center for every depth -- the whole point of the demo.
_, _, forward1 = camera_basis(ray_dir)
F1, HALF_W1, HALF_H1, SHEAR1 = 1.1, 0.55, 0.42, 0.35
center1 = C1 + F1 * forward1
corners1 = sheared_quad(center1, HALF_W1, HALF_H1, SHEAR1)

# Camera 2's image plane: oriented to look generally back along the ray.
# Its shear runs the opposite way (right edge higher), as in the reference.
# Since corners2 ends up built purely from X_DIR/Y_DIR offsets from its
# center, the drawn (schematic) plane 2 is EXACTLY a flat, constant-Z plane
# -- not the true perspective image plane. Everything below works directly
# in that plane's own geometry (not camera 2's real right/up basis), which
# is what makes x2 and the epipolar line land consistently on the shape
# actually drawn, with the correct slope.
TARGET = C1 + 3.5 * ray_dir
_, _, forward2 = camera_basis(TARGET - C2)
SHEAR2 = -0.35
HALF_W2, HALF_H2 = HALF_W1, HALF_H1  # same size as plane 1, not derived independently


def _project_at(Xw, plane_z):
    """Where the sightline C2->Xw crosses the flat plane Z = plane_z."""
    direction = Xw - C2
    s = (plane_z - C2[2]) / direction[2]
    return C2 + s * direction


# Solve for F2 (plane 2's distance from C2 along forward2) so the full swept
# range of x2 over DEPTH_MIN..DEPTH_MAX fits inside HALF_W2 with a margin --
# instead of guessing F2 and letting the plane's size/shape fall out of that
# guess, which previously made plane 2 end up much bigger than plane 1.
_MARGIN = 1.15
_z_ref = (C2 + 1.0 * forward2)[2]  # reference plane at F2 = 1.0, to measure the natural spread
_ref_min = _project_at(C1 + DEPTH_MIN * ray_dir, _z_ref)
_ref_max = _project_at(C1 + DEPTH_MAX * ray_dir, _z_ref)
_ref_half_w = abs(_ref_min[0] - _ref_max[0]) / 2  # x is the binding (much larger) dimension here
F2 = (HALF_W2 / _MARGIN) / _ref_half_w
_PLANE2_Z = (C2 + F2 * forward2)[2]


def project_to_plane2(Xw):
    return _project_at(Xw, _PLANE2_Z)


_x2_min = project_to_plane2(C1 + DEPTH_MIN * ray_dir)
_x2_max = project_to_plane2(C1 + DEPTH_MAX * ray_dir)
center2 = (_x2_min + _x2_max) / 2
corners2 = sheared_quad(center2, HALF_W2, HALF_H2, SHEAR2)


def _clip_to_quad(mid, direction, half_w, half_h, shear):
    """t-range (along mid + t*direction) that lies inside the sheared quad in
    the same (u, v) = (X_DIR, Y_DIR) offsets from center2 used by sheared_quad."""
    u0, v0 = mid
    du, dv = direction
    ts = [(-half_w - u0) / du, (half_w - u0) / du] if du != 0 else [-1e6, 1e6]
    t_lo, t_hi = min(ts), max(ts)

    coef = dv + shear * du / (2 * half_w)
    upper_rhs = half_h - v0 - shear * u0 / (2 * half_w)
    lower_rhs = -half_h - v0 - shear * u0 / (2 * half_w)
    if coef > 0:
        t_lo, t_hi = max(t_lo, lower_rhs / coef), min(t_hi, upper_rhs / coef)
    elif coef < 0:
        t_lo, t_hi = max(t_lo, upper_rhs / coef), min(t_hi, lower_rhs / coef)
    return t_lo, t_hi


# The epipolar line is the intersection of the (fixed) epipolar plane --
# spanned by C1, C2, and the ray -- with plane 2's own flat Z = const plane.
# Two actual planes intersecting in 3D, not a basis-mismatched approximation,
# so the slope is guaranteed right and x2 (above) always lands exactly on it.
_epi_normal = np.cross(C2 - C1, ray_dir)
_nx, _ny, _nz = _epi_normal
if abs(_ny) > abs(_nx):
    _pt0 = np.array([C1[0], C1[1] - _nz * (_PLANE2_Z - C1[2]) / _ny, _PLANE2_Z])
else:
    _pt0 = np.array([C1[0] - _nz * (_PLANE2_Z - C1[2]) / _nx, C1[1], _PLANE2_Z])
_line_dir3d = np.array([-_ny, _nx, 0.0])
_line_dir3d = _line_dir3d / np.linalg.norm(_line_dir3d)

_mid_uv = (_pt0[0] - center2[0], _pt0[1] - center2[1])
_dir_uv = (_line_dir3d[0], _line_dir3d[1])
_t_lo, _t_hi = _clip_to_quad(_mid_uv, _dir_uv, HALF_W2, HALF_H2, SHEAR2)
line_pt_a = _pt0 + _t_lo * _line_dir3d
line_pt_b = _pt0 + _t_hi * _line_dir3d


def render(depth, trail_depths, final=False):
    X = C1 + depth * ray_dir
    x2 = project_to_plane2(X)

    fig = plt.figure(figsize=(7.4, 6.4), dpi=120)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # The (fixed) epipolar plane, spanned by C1, C2, and the ray.
    plane = Poly3DCollection([[V(C1), V(C2), V(FAR_POINT)]], alpha=0.12, facecolor=ACTIVE_COLOR)
    ax.add_collection3d(plane)
    ax.plot(*zip(V(C1), V(C2)), c="gray", linestyle="--", linewidth=1)

    # Image plane 1: x1 is fixed at its center for every depth.
    ax.add_collection3d(Poly3DCollection([[V(p) for p in corners1]], alpha=0.25, facecolor=PLANE1_COLOR,
                                           edgecolor=PLANE1_COLOR, linewidths=1.2))
    ax.scatter(*V(center1), c=NEUTRAL_COLOR, s=55, zorder=6, edgecolors="white", linewidths=0.8)
    ax.text(*V(center1 + 0.1 * Y_DIR + 0.06 * X_DIR), "$x_1$ (fixed)", fontsize=8.5, color=NEUTRAL_COLOR)

    # Image plane 2: the epipolar line is this plane's edge-on view of the
    # epipolar plane -- drawn directly on the plane, not off to the side.
    ax.add_collection3d(Poly3DCollection([[V(p) for p in corners2]], alpha=0.25, facecolor=PLANE2_COLOR,
                                           edgecolor=PLANE2_COLOR, linewidths=1.2))
    ax.plot(*zip(V(line_pt_a), V(line_pt_b)), c=MATCH_COLOR, linewidth=1.6, zorder=4)
    for d in trail_depths:
        ax.scatter(*V(project_to_plane2(C1 + d * ray_dir)), c="#bbbbbb", s=14, zorder=4)
    marker_color = MATCH_COLOR if final else ACTIVE_COLOR
    ax.scatter(*V(x2), c=marker_color, s=85, zorder=7, edgecolors="black", linewidths=0.7)

    # The ray itself, and camera 2's sightline to the current X.
    for d in trail_depths:
        ax.scatter(*V(C1 + d * ray_dir), c="#bbbbbb", s=14, zorder=2)
    ax.plot(*zip(V(C1), V(X)), c=PLANE1_COLOR, linewidth=1.3)
    ax.plot(*zip(V(C2), V(X)), c=PLANE2_COLOR, linewidth=1.3)
    ax.scatter(*V(X), c=marker_color, s=90, zorder=5, edgecolors="black", linewidths=0.7)

    ax.scatter(*V(C1), c=PLANE1_COLOR, s=55)
    ax.text(*V(C1 - np.array([0.12, 0.0, 0.0])), "$C_1$", fontsize=9, ha="right")
    ax.scatter(*V(C2), c=PLANE2_COLOR, s=55)
    ax.text(*V(C2), "  $C_2$", fontsize=9)

    ax.set_axis_off()
    ax.set_xlim3d(-0.6, 3.2); ax.set_ylim3d(-0.7, 6.0); ax.set_zlim3d(-0.3, 1.7)
    ax.view_init(elev=10, azim=-70)

    fig.suptitle(f"camera 1's ray at depth {depth:4.1f}  →  one point on the epipolar line in image 2",
                 fontsize=11.5, color=NEUTRAL_COLOR, fontweight="bold")
    fig.subplots_adjust(top=2.05, bottom=-1.05, left=-0.15, right=1.15)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_STEP = 3
HOLD_END = 44

depths = np.linspace(DEPTH_MIN, DEPTH_MAX, 60)
trail = []
for d in depths:
    frames += [render(d, trail)] * HOLD_STEP
    if len(trail) < 3 and (len(trail) == 0 or d - trail[-1] > 1.4):
        trail = trail + [d]

frames += [render(depths[-1], trail, final=True)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=55,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
