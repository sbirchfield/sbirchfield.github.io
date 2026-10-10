"""Generate hough_voting_demo.gif: an animated walkthrough of the lesson 12
notebook's "The accumulator: voting for lines" cell. Instead of only showing
the final accumulator (built from every edge pixel at once), this adds edge
points one at a time, each casting its full (rho, theta) sinusoid into the
accumulator, so the viewer watches the votes pile up at the one bin that
every collinear point agrees on.

Uses the same synthetic line and accumulator setup as the notebook cell.

Run from anywhere:
    python make_hough_voting_demo.py
Output:
    hough_voting_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "hough_voting_demo.gif")

ACTIVE_COLOR = "#e07b00"
DONE_COLOR = "#3a6fb0"
MATCH_COLOR = "#2e9e44"
TRUE_COLOR = "#00c8c8"

SIZE = 300
P1, P2 = (30, 270), (270, 30)  # same x + y = 300 line as the notebook

theta_true = np.radians(45)
rho_true = 300 * np.cos(theta_true)

N_POINTS = 24
xs = np.linspace(P1[0], P2[0], N_POINTS).astype(int)
ys = np.linspace(P1[1], P2[1], N_POINTS).astype(int)

theta_bins = np.linspace(0, np.pi, 180, endpoint=False)
max_rho = int(np.hypot(SIZE, SIZE))
n_rho_bins = 2 * max_rho


def render(n_added, final_hold=False):
    accumulator = np.zeros((n_rho_bins, len(theta_bins)), dtype=np.int32)
    for i in range(n_added):
        rho_vals = (xs[i] * np.cos(theta_bins) + ys[i] * np.sin(theta_bins)).astype(int) + max_rho
        accumulator[rho_vals, np.arange(len(theta_bins))] += 1

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), dpi=120)
    fig.subplots_adjust(top=0.82, bottom=0.12, wspace=0.28)
    ax_img, ax_acc = axes

    ax_img.set_xlim(0, SIZE)
    ax_img.set_ylim(SIZE, 0)
    ax_img.set_aspect("equal")
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_visible(False)
    if n_added > 1:
        ax_img.scatter(xs[:n_added - 1], ys[:n_added - 1], c=DONE_COLOR, s=28, zorder=2)
    if 0 < n_added <= N_POINTS:
        ax_img.scatter([xs[n_added - 1]], [ys[n_added - 1]], c=ACTIVE_COLOR, s=60, zorder=3,
                         edgecolors="white", linewidths=1.2)
    ax_img.set_title(f"image space: {n_added} / {N_POINTS} edge points added", fontsize=10.5)

    ax_acc.imshow(np.log1p(accumulator), cmap="hot", aspect="auto",
                   extent=[0, 180, -max_rho, max_rho], origin="lower")
    if 0 < n_added <= N_POINTS and not final_hold:
        rho_curve = xs[n_added - 1] * np.cos(theta_bins) + ys[n_added - 1] * np.sin(theta_bins)
        ax_acc.plot(np.degrees(theta_bins), rho_curve, color=ACTIVE_COLOR, linewidth=1.8)
    ax_acc.scatter([np.degrees(theta_true)], [rho_true], edgecolor=TRUE_COLOR, facecolor="none",
                    s=160, linewidth=1.6, label="true (rho, theta)", zorder=4)

    if final_hold:
        peak_idx, peak_theta_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        peak_rho = peak_idx - max_rho
        peak_votes = int(accumulator.max())
        ax_acc.scatter([np.degrees(theta_bins[peak_theta_idx])], [peak_rho], marker="*",
                        color=MATCH_COLOR, s=220, zorder=5, edgecolors="white", linewidths=0.8)
        title = f"peak bin = {peak_votes} votes, matches the true line"
        color = MATCH_COLOR
    else:
        title = "accumulator (rho, theta), log-scaled"
        color = "#222222"
    ax_acc.legend(fontsize=7.5, loc="upper right")
    ax_acc.set_xlabel("theta (degrees)", fontsize=9.5)
    ax_acc.set_ylabel("rho", fontsize=9.5)
    ax_acc.set_title(title, fontsize=10.5, color=color)

    suptitle = ("every point votes for its whole sinusoid -- collinear points agree on one bin"
                if not final_hold else "all votes cast: the peak bin recovers the line's (rho, theta)")
    fig.suptitle(suptitle, fontsize=12.5, color=color if final_hold else "#222222",
                  fontweight="bold" if final_hold else "normal")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
HOLD_INTRO = 10
HOLD_POINT = 5
HOLD_DONE = 16
HOLD_END = 44

frames += [render(0)] * HOLD_INTRO
for n in range(1, N_POINTS + 1):
    frames += [render(n)] * HOLD_POINT
frames += [render(N_POINTS)] * HOLD_DONE
frames += [render(N_POINTS, final_hold=True)] * HOLD_END

frames[0].save(
    OUT_GIF,
    save_all=True,
    append_images=frames[1:],
    duration=90,
    loop=0,
)
print(f"Saved {OUT_GIF} ({len(frames)} frames, {N_POINTS} points)")
