"""Evaluate temporal variants V0-V3 on REVIDE (indoor test clips). Resumable per (clip,variant,alpha).

Usage: python -m eval.revide_runner [--width 640] [--alphas 0.5 0.7 0.9] [--clips C005 ...]
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np

from dcp import load_config
from dcpvid import dehaze_video, VARIANTS
from eval.data import data_cfg, read_rgb
from eval.metrics import psnr, ssim
from eval.temporal_metrics import ClipReference

OUT = "results/video_ablation.csv"
FIELDS = ["clip", "variant", "alpha", "frames", "psnr", "ssim", "flicker_index", "roi_var", "fps"]


def load_clip(split, clip, width):
    root = data_cfg()["revide_root"]
    hz = sorted(glob.glob(f"{root}/{split}/hazy/{clip}/*.JPG"))
    gt = [p.replace("/hazy/", "/gt/") for p in hz]

    def rd(p):
        im = read_rgb(p)
        h = int(round(im.shape[0] * width / im.shape[1]))
        return cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA)
    return [rd(p) for p in hz], [rd(p) for p in gt]


def clips(split="Test"):
    return sorted(os.listdir(os.path.join(data_cfg()["revide_root"], split, "hazy")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="Test")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.5, 0.7, 0.9])
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {(r["clip"], r["variant"], r["alpha"]) for r in csv.DictReader(f)}
    cfg = load_config()
    for clip in a.clips or clips(a.split):
        jobs = [(v, al) for v in VARIANTS for al in ([0.0] if v == "V0" else a.alphas)
                if (clip, v, str(al)) not in done]
        if not jobs:
            continue
        hazy, gt = load_clip(a.split, clip, a.width)
        ref = ClipReference(gt)
        for v, al in jobs:
            out = dehaze_video(hazy, v, al, cfg)
            J = out["J"]
            row = dict(clip=clip, variant=v, alpha=al, frames=len(J),
                       psnr=np.mean([psnr(j, g) for j, g in zip(J, gt)]),
                       ssim=np.mean([ssim(j, g) for j, g in zip(J, gt)]),
                       flicker_index=ref.flicker_index(J), roi_var=ref.roi_variance(J), fps=out["fps"])
            new = not os.path.exists(a.out)
            with open(a.out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
            print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items()}, flush=True)


if __name__ == "__main__":
    main()
