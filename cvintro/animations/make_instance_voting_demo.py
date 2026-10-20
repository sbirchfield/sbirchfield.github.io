"""Generate instance_voting_demo.gif: an animated companion to the lesson 43
notebook's "Recovering instances: vote for the center, then cluster (Hough,
again)" section. Animates the offset-voting + clustering algorithm itself, on
the first held-out test scene (the leftmost panel of the notebook's static
3x4 comparison figure): foreground pixels cast votes into an accumulator,
two tight clusters emerge, then the same greedy peak-pick/suppress loop as
Lesson 40's NMS recovers the two instance centers.

Reuses the notebook's own make_scene/InstanceNet/cluster_instances setup
verbatim (data rng seed=17, model seed=0, 300 epochs, lr=0.01), so the
final instance assignment matches the book exactly.

Run from anywhere:
    python make_instance_voting_demo.py
Output:
    instance_voting_demo.gif  (written next to this script)
"""
import io
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_GIF = os.path.join(OUT_DIR, "instance_voting_demo.gif")

TEXT_COLOR = "#222222"
FG_COLOR = "#2e9e44"
PEAK_COLOR = "#d62728"      # the vote peak currently being picked (red dashed,
                            # matching the NMS "pick" convention elsewhere)
SUPPRESS_COLOR = "#9e9e9e"  # suppressed neighborhood around a picked peak

SIZE = 32


def make_scene(rng, size=SIZE, n_circles=2, r=5):
    scene = np.zeros((size, size), dtype=np.float32)
    sem_mask = np.zeros((size, size), dtype=np.int64)
    inst_mask = np.zeros((size, size), dtype=np.int64)
    intensity = {0: 0.85, 1: 1.0}
    centers = []
    for k in range(n_circles):
        if k == 0:
            cx, cy = rng.integers(r + 8, size - r - 8), rng.integers(r + 8, size - r - 8)
        else:
            px, py = centers[-1]
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(6, 9)
            cx = int(np.clip(px + dist * np.cos(angle), r, size - r - 1))
            cy = int(np.clip(py + dist * np.sin(angle), r, size - r - 1))
        centers.append((cx, cy))
        yy, xx = np.mgrid[0:size, 0:size]
        m = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2
        scene[m] = intensity[k]
        sem_mask[m] = 1
        inst_mask[m] = k + 1
    scene = np.clip(scene + rng.normal(0, 0.05, scene.shape), 0, 1).astype(np.float32)
    return scene, sem_mask, inst_mask, centers


rng = np.random.default_rng(17)
N = 300
scenes, sem_masks, inst_masks, all_centers = [], [], [], []
for _ in range(N):
    s, sm, im, c = make_scene(rng)
    scenes.append(s); sem_masks.append(sm); inst_masks.append(im); all_centers.append(c)
scenes = np.array(scenes, dtype=np.float32)
sem_masks = np.array(sem_masks, dtype=np.int64)

yy, xx = np.mgrid[0:SIZE, 0:SIZE]
offset_targets = np.zeros((N, 2, SIZE, SIZE), dtype=np.float32)
for i in range(N):
    im = inst_masks[i]
    for k, (cx, cy) in enumerate(all_centers[i]):
        m = im == (k + 1)
        offset_targets[i, 0][m] = (cx - xx[m]) / SIZE
        offset_targets[i, 1][m] = (cy - yy[m]) / SIZE

split = int(0.85 * N)
Xtr, Str, Otr = scenes[:split], sem_masks[:split], offset_targets[:split]
Xte, Ste, Ote = scenes[split:], sem_masks[split:], offset_targets[split:]
inst_te, centers_te = inst_masks[split:], all_centers[split:]


class InstanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = nn.Sequential(nn.Conv2d(32 + 16, 16, 3, padding=1), nn.ReLU())
        self.sem_head = nn.Conv2d(16, 2, 1)
        self.offset_head = nn.Conv2d(16, 2, 1)

    def forward(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(self.pool(f1))
        d1 = self.dec1(torch.cat([self.up(f2), f1], dim=1))
        return self.sem_head(d1), self.offset_head(d1)


print("training the instance network...")
torch.manual_seed(0)
model = InstanceNet()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
Xt = torch.tensor(Xtr).unsqueeze(1); St = torch.tensor(Str); Ot = torch.tensor(Otr)
fg_mask_t = (St == 1).unsqueeze(1).float()
for _ in range(300):
    opt.zero_grad()
    sem_logits, offset_pred = model(Xt)
    sem_loss = F.cross_entropy(sem_logits, St)
    offset_loss = (F.mse_loss(offset_pred, Ot, reduction="none") * fg_mask_t).sum() / fg_mask_t.sum().clamp(min=1)
    (sem_loss + 2.0 * offset_loss).backward()
    opt.step()

with torch.no_grad():
    sem_logits_te, offset_pred_te = model(torch.tensor(Xte).unsqueeze(1))
    sem_preds = sem_logits_te.argmax(1).numpy()
    offset_preds = offset_pred_te.numpy()
print(f"semantic pixel accuracy: {(sem_preds == Ste).mean():.1%}")

# ---------------------------------------------------------------------------
# The exact same scene as the leftmost panel of the notebook's static 3x4
# figure (afe7a96d): index 0 of the held-out test set.
# ---------------------------------------------------------------------------
IDX = 0
scene = Xte[IDX]
binary_mask = sem_preds[IDX] == 1
offset_pred = offset_preds[IDX]
true_inst = inst_te[IDX]

PEAK_DIST = 4
VOTE_THRESH = 3

ys, xs = np.where(binary_mask)
pred_cx = xs + offset_pred[0][ys, xs] * SIZE
pred_cy = ys + offset_pred[1][ys, xs] * SIZE

# Shuffle vote order (not spatial raster order) so the progressive reveal
# below grows both instance clusters roughly simultaneously.
vote_order = np.random.default_rng(0).permutation(len(xs))
pred_cx, pred_cy = pred_cx[vote_order], pred_cy[vote_order]

votes_full = np.zeros((SIZE, SIZE), dtype=np.float32)
for cx, cy in zip(pred_cx, pred_cy):
    cxi = int(np.clip(round(cx), 0, SIZE - 1)); cyi = int(np.clip(round(cy), 0, SIZE - 1))
    votes_full[cyi, cxi] += 1

# Trace the exact cluster_instances() loop, recording one state per beat.
votes_work = votes_full.copy()
peaks = []
beats = []
for r in range(6):
    idx = np.unravel_index(votes_work.argmax(), votes_work.shape)
    val = votes_work[idx]
    if val < VOTE_THRESH:
        beats.append(dict(phase="stop", round=r + 1, votes=votes_work.copy(), peaks=list(peaks), val=val))
        break
    peaks.append((idx[1], idx[0]))
    beats.append(dict(phase="pick", round=r + 1, votes=votes_work.copy(), peaks=list(peaks), val=val,
                       peak=(idx[1], idx[0])))
    y0, y1 = max(0, idx[0] - PEAK_DIST), min(SIZE, idx[0] + PEAK_DIST + 1)
    x0, x1 = max(0, idx[1] - PEAK_DIST), min(SIZE, idx[1] + PEAK_DIST + 1)
    votes_work[y0:y1, x0:x1] = 0
    beats.append(dict(phase="suppress", round=r + 1, votes=votes_work.copy(), peaks=list(peaks),
                       peak=(idx[1], idx[0]), box=(x0, y0, x1, y1)))

inst_pred = np.zeros_like(binary_mask, dtype=np.int64)
if peaks:
    peaks_arr = np.array(peaks)
    for y, x in zip(ys, xs):
        dists = (peaks_arr[:, 0] - x) ** 2 + (peaks_arr[:, 1] - y) ** 2
        inst_pred[y, x] = np.argmin(dists) + 1
print(f"recovered {len(peaks)} instance(s) for this scene (true count: "
      f"{len(set(true_inst.ravel()) - {0})})")

# "magma" (dark purple -> bright yellow) rather than "hot": "hot" already uses
# red/orange for high-vote cells, which visually clashes with the red "X"
# peak marker and makes it hard to tell marker from heat. magma's palette
# stays clearly distinct from the red/black/white markers at every vote count.
VOTES_CMAP = plt.cm.magma.copy()
VOTES_CMAP.set_bad(color="white")

FOUND_PEAK_COLOR = "#1f6fb4"  # previously-found peak, distinct from the
                              # red "currently being picked" peak


def draw_votes_panel(ax, votes, peaks_so_far, highlight_peak=None, suppress_box=None, vmax=None):
    disp = np.ma.masked_equal(votes, 0)
    ax.imshow(disp, cmap=VOTES_CMAP, vmin=0, vmax=vmax or max(1, votes.max()))
    for (px, py) in peaks_so_far:
        if highlight_peak is not None and (px, py) == highlight_peak:
            continue
        ax.plot(px, py, marker="x", color="black", markersize=12, mew=3.6, zorder=4)
        ax.plot(px, py, marker="x", color=FOUND_PEAK_COLOR, markersize=8, mew=2, zorder=5)
    if highlight_peak is not None:
        px, py = highlight_peak
        ax.plot(px, py, marker="x", color="black", markersize=15, mew=4.4, zorder=6)
        ax.plot(px, py, marker="x", color=PEAK_COLOR, markersize=10, mew=2.4, zorder=7)
    if suppress_box is not None:
        x0, y0, x1, y1 = suppress_box
        ax.add_patch(patches.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, edgecolor=SUPPRESS_COLOR,
                                        facecolor="none", linewidth=2, linestyle=":"))
    ax.set_xlim(-0.5, SIZE - 0.5); ax.set_ylim(SIZE - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])


# Proxy legend handles -- shown underneath the vote-accumulator panel
# whenever a peak marker or suppression box is on screen, so the viewer
# doesn't have to guess what the X's and dashed box mean.
LEGEND_HANDLES = [
    plt.Line2D([0], [0], marker="x", color=PEAK_COLOR, markeredgecolor=PEAK_COLOR,
               linestyle="none", markersize=9, mew=2.4, label="peak picked this round"),
    plt.Line2D([0], [0], marker="x", color=FOUND_PEAK_COLOR, markeredgecolor=FOUND_PEAK_COLOR,
               linestyle="none", markersize=9, mew=2, label="peak found earlier"),
    patches.Patch(facecolor="none", edgecolor=SUPPRESS_COLOR, linestyle=":", linewidth=2,
                  label="suppressed neighborhood"),
]


def render(scene_panel_title, votes=None, peaks_so_far=None, highlight_peak=None, suppress_box=None,
           final=False, subtitle=""):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.9), dpi=120)
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.03, right=0.97, wspace=0.15)

    ax0 = axes[0]
    if not final:
        ax0.imshow(scene, cmap="gray", vmin=0, vmax=1)
        fg = np.ma.masked_where(~binary_mask, binary_mask)
        ax0.imshow(fg, cmap="Greens", vmin=0, vmax=1.4, alpha=0.55)
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_title(scene_panel_title, fontsize=9.5, color=TEXT_COLOR)
    else:
        ax0.imshow(true_inst, cmap="viridis")
        ax0.axis("off")
        ax0.set_title("ground truth instances", fontsize=9.5, color=TEXT_COLOR)

    ax1 = axes[1]
    if not final:
        draw_votes_panel(ax1, votes, peaks_so_far or [], highlight_peak, suppress_box, vmax=votes_full.max())
        ax1.set_title("vote accumulator" if votes is not None else "", fontsize=9.5, color=TEXT_COLOR)
        if highlight_peak is not None or peaks_so_far:
            ax1.legend(handles=LEGEND_HANDLES, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                       fontsize=7.6, frameon=False, handletextpad=0.5, labelspacing=0.3)
    else:
        ax1.imshow(inst_pred, cmap="viridis")
        ax1.axis("off")
        ax1.set_title("recovered via offset voting", fontsize=9.5, color=TEXT_COLOR)

    fig.suptitle("vote for the center, then cluster", fontsize=13, fontweight="bold", color=TEXT_COLOR, y=0.97)
    if subtitle:
        fig.text(0.5, 0.86, subtitle, ha="center", fontsize=10, color=TEXT_COLOR, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


frames = []

# Beat 1: intro -- scene + predicted foreground mask.
frames += [render("scene + predicted foreground mask", votes=np.zeros((SIZE, SIZE)), peaks_so_far=[])] * 40

# Beat 2: progressive vote build-up.
N_CHUNKS = 10
chunk_size = max(1, len(pred_cx) // N_CHUNKS)
votes_running = np.zeros((SIZE, SIZE), dtype=np.float32)
for c in range(N_CHUNKS):
    lo, hi = c * chunk_size, (c + 1) * chunk_size if c < N_CHUNKS - 1 else len(pred_cx)
    for cx, cy in zip(pred_cx[lo:hi], pred_cy[lo:hi]):
        cxi = int(np.clip(round(cx), 0, SIZE - 1)); cyi = int(np.clip(round(cy), 0, SIZE - 1))
        votes_running[cyi, cxi] += 1
    frames += [render("foreground pixels casting votes", votes=votes_running.copy(), peaks_so_far=[],
                       subtitle=f"votes cast: {hi}/{len(pred_cx)}")] * 6
frames += [render("foreground pixels casting votes", votes=votes_running.copy(), peaks_so_far=[],
                   subtitle=f"votes cast: {len(pred_cx)}/{len(pred_cx)}")] * 30

# Beat 3: replay the real greedy peak-pick/suppress loop.
peaks_shown = []
for b in beats:
    if b["phase"] == "pick":
        frames += [render("greedily picking the highest-voted peak", votes=b["votes"], peaks_so_far=peaks_shown,
                           highlight_peak=b["peak"], subtitle=f"round {b['round']}: peak has {int(b['val'])} votes"
                           )] * 30
    elif b["phase"] == "suppress":
        frames += [render("suppressing the neighborhood around that peak", votes=b["votes"],
                           peaks_so_far=peaks_shown, highlight_peak=b["peak"], suppress_box=b["box"],
                           subtitle=f"round {b['round']}: neighborhood zeroed out")] * 30
        peaks_shown = list(b["peaks"])
    elif b["phase"] == "stop":
        frames += [render("remaining peak below vote_thresh -- stop", votes=b["votes"], peaks_so_far=peaks_shown,
                           subtitle=f"round {b['round']}: best remaining peak has only "
                                    f"{int(b['val'])} vote(s) < {VOTE_THRESH}")] * 35

# Beat 4: final instance assignment vs. ground truth.
frames += [render("", final=True)] * 90

frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=50, loop=0)
print(f"Saved {OUT_GIF} ({len(frames)} frames)")
