"""Generate fp16_underflow_demo.gif: an animated companion to the lesson 44
notebook's "Floating-point precision: fp32 vs. fp16" section. Reveals the
notebook's exact eight values, largest magnitude to smallest, one at a time,
color-coding which survive the cast to fp16 intact, which survive but lose
precision, and which underflow to exactly 0.0.

Reuses the notebook's own small_values tensor and .half() cast verbatim, so
the outcome for every value is computed live, not hardcoded.

Run from anywhere:
    python make_fp16_underflow_demo.py
Output:
    fp16_underflow_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "fp16_underflow_demo.gif")

TEXT_COLOR = "#222222"
INTACT_COLOR = "#2e9e44"
IMPRECISE_COLOR = "#f2994a"
ZERO_COLOR = "#d62728"
FLOOR_COLOR = "#888888"

small_values = torch.tensor([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9], dtype=torch.float32)
small_fp16 = small_values.half()

FP16_NORMAL_FLOOR = torch.finfo(torch.float16).tiny      # smallest normal, ~6.1e-5
FP16_SUBNORMAL_FLOOR = 2 ** -24                            # smallest positive subnormal, ~6e-8

rows = []
for v32, v16 in zip(small_values.tolist(), small_fp16.tolist()):
    # Match the notebook's own narrative: "intact" means it prints the same
    # at 1-decimal scientific precision (its actual granularity at this
    # magnitude), not a stricter floating-point-exact comparison.
    if v16 == 0.0:
        status = "zero"
    elif f"{v16:.1e}" == f"{v32:.1e}":
        status = "intact"
    else:
        status = "imprecise"
    rows.append(dict(v32=v32, v16=v16, status=status))

print("fp32 value   cast to fp16   status")
for r in rows:
    print(f"{r['v32']:>10.1e}   {r['v16']:>11.2e}   {r['status']}")

STATUS_COLOR = {"intact": INTACT_COLOR, "imprecise": IMPRECISE_COLOR, "zero": ZERO_COLOR}
STATUS_LABEL = {"intact": "survives intact", "imprecise": "survives, loses precision", "zero": "underflows to 0.0"}

Y_MIN, Y_MAX = -10.5, 0.3  # log10 axis range covering 1e-9 .. 1e-2, with headroom for labels


def render(n_shown):
    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=120)
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.14, right=0.97)

    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{r['v32']:.0e}" for r in rows], fontsize=8.5)
    ax.set_ylabel("magnitude (log10)", fontsize=9.5, color=TEXT_COLOR)
    ax.set_xlabel("fp32 value", fontsize=9.5, color=TEXT_COLOR)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    floor_normal = np.log10(FP16_NORMAL_FLOOR)
    floor_subnormal = np.log10(FP16_SUBNORMAL_FLOOR)
    ax.axhline(floor_normal, color=FLOOR_COLOR, linewidth=1.4, linestyle="--", zorder=1)
    ax.text(len(rows) - 0.5, floor_normal + 0.15, "fp16 normal floor (~6.1e-5)", ha="right",
            fontsize=7.6, color=FLOOR_COLOR)
    ax.axhline(floor_subnormal, color=FLOOR_COLOR, linewidth=1.0, linestyle=":", zorder=1)
    ax.text(len(rows) - 0.5, floor_subnormal + 0.15, "fp16 subnormal floor (~6e-8)", ha="right",
            fontsize=7.6, color=FLOOR_COLOR)

    bar_w = 0.32
    for i, r in enumerate(rows[:n_shown]):
        color = STATUS_COLOR[r["status"]]
        log_v32 = np.log10(r["v32"])
        ax.bar(i - bar_w / 2 - 0.02, log_v32 - Y_MIN, width=bar_w, bottom=Y_MIN,
               color="#9fb8d9", edgecolor="none", zorder=2)
        if r["v16"] > 0:
            log_v16 = np.log10(r["v16"])
            ax.bar(i + bar_w / 2 + 0.02, log_v16 - Y_MIN, width=bar_w, bottom=Y_MIN,
                   color=color, edgecolor="none", zorder=2)
        else:
            ax.scatter([i + bar_w / 2 + 0.02], [Y_MIN + 0.12], marker="x", s=90, color=color,
                       linewidth=2.4, zorder=3)
        ax.text(i, log_v32 + 0.35, STATUS_LABEL[r["status"]], ha="center", fontsize=6.6,
                color=color, fontweight="bold", rotation=90, va="bottom")

    ax.bar([-5], [0], color="#9fb8d9", label="fp32 (always intact)")
    ax.bar([-5], [0], color=INTACT_COLOR, label="fp16: intact")
    ax.bar([-5], [0], color=IMPRECISE_COLOR, label="fp16: loses precision")
    ax.bar([-5], [0], color=ZERO_COLOR, label="fp16: underflows to 0.0")
    ax.legend(loc="upper right", fontsize=7.6, frameon=False, ncol=1)

    fig.suptitle("casting small values from fp32 to fp16", fontsize=13, fontweight="bold",
                 color=TEXT_COLOR, y=0.97)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []
for n in range(1, len(rows) + 1):
    hold = 10 if n < len(rows) else 90
    frames += [render(n)] * hold

frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=55, loop=0)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
