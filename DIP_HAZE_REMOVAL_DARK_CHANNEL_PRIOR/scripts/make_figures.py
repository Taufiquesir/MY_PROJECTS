"""Regenerate figures from results/*.csv."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("figures", exist_ok=True)


def rd(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def sky_scatter():
    r = rd("results/failure_cases.csv")
    x = np.array([float(a["sky_frac"]) for a in r])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(x, [float(a["psnr_base"]) for a in r], s=12, label="baseline DCP", alpha=.7)
    ax.scatter(x, [float(a["psnr_mit"]) for a in r], s=12, label="+ sky t-floor", alpha=.7)
    ax.set_xlabel("sky fraction (GT image)"); ax.set_ylabel("PSNR (dB)"); ax.legend()
    ax.set_title("RESIDE SOTS outdoor: PSNR vs sky fraction")
    fig.tight_layout(); fig.savefig("figures/sky_psnr_scatter.png", dpi=150)


def video_bars():
    r = rd("results/video_ablation.csv")
    labels, vals = [], []
    for v, al in [("V0", "0.0"), ("V1", "0.7"), ("V2", "0.7"), ("V3", "0.7")]:
        s = [float(a["flicker_index"]) for a in r if a["variant"] == v and a["alpha"] == al]
        labels.append(v); vals.append(np.mean(s))
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.bar(labels, vals); ax.axhline(1.0, ls="--", c="k", lw=1)
    ax.set_ylabel("flicker index (GT = 1)"); ax.set_title("REVIDE test, alpha=0.7 (mean of 6 clips)")
    fig.tight_layout(); fig.savefig("figures/video_flicker.png", dpi=150)


def stages(split="outdoor", name="0001_0.8_0.2.jpg"):
    """Every pipeline stage for one SOTS image (Deliverable 1: intermediates exposed)."""
    import sys
    sys.path.insert(0, ".")
    from dcp import dehaze, load_config
    from eval.data import data_cfg, read_rgb
    root = os.path.join(data_cfg()["sots_root"], split)
    I = read_rgb(os.path.join(root, "hazy", name)); G = read_rgb(os.path.join(root, "gt", name.split("_")[0] + ".png"))
    o = dehaze(I, load_config())
    panels = [("hazy I", I), ("dark channel", o["dark"]), ("raw t", o["t_raw"]), ("refined t (guided)", o["t_ref"]),
              ("recovered J", o["J"]), ("ground truth", G)]
    fig, ax = plt.subplots(2, 3, figsize=(11, 6))
    for a, (t, im) in zip(ax.ravel(), panels):
        a.imshow(im, cmap=None if im.ndim == 3 else "gray", vmin=0, vmax=1); a.set_title(t); a.axis("off")
    fig.tight_layout(); fig.savefig("figures/single_image_stages.png", dpi=130)


if __name__ == "__main__":
    sky_scatter(); video_bars(); stages()
