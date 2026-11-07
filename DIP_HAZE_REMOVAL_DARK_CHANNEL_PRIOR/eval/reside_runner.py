"""Run the single-image pipeline over RESIDE SOTS and write per-image metrics (CSV).

Usage:
  python -m eval.reside_runner --split indoor --n 50 --set refine.method=guided --out results/x.csv
"""
import argparse
import csv
import os
import time
from multiprocessing import Pool

import numpy as np

from dcp import dehaze, load_config
from eval.data import data_cfg, read_rgb
from eval.metrics import psnr, ssim, ciede2000


def list_pairs(split, n=None, seed=0):
    root = os.path.join(data_cfg()["sots_root"], split)
    hazy = sorted(os.listdir(os.path.join(root, "hazy")))
    if n and n < len(hazy):                      # fixed random subset -> same images across all configs
        rng = np.random.default_rng(seed)
        hazy = [hazy[i] for i in sorted(rng.choice(len(hazy), n, replace=False))]
    pairs = []
    for f in hazy:
        gt = [g for g in os.listdir(os.path.join(root, "gt")) if g.split(".")[0] == f.split("_")[0]][0]
        pairs.append((os.path.join(root, "hazy", f), os.path.join(root, "gt", gt)))
    return pairs


def _run_one(args):
    (hp, gp), cfg = args
    I, G = read_rgb(hp), read_rgb(gp)
    t0 = time.perf_counter()
    out = dehaze(I, cfg)
    dt = time.perf_counter() - t0
    J = out["J"]
    return dict(image=os.path.basename(hp), psnr=psnr(J, G), ssim=ssim(J, G),
                ciede2000=ciede2000(J, G), seconds=dt)


def evaluate(split, cfg, n=None, workers=4):
    """Returns list of per-image dicts. Timing is measured inside the worker, per image."""
    jobs = [(p, cfg) for p in list_pairs(split, n)]
    with Pool(workers) as pool:
        return pool.map(_run_one, jobs, chunksize=1)


def summarize(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in ["psnr", "ssim", "ciede2000", "seconds"]} | {"n": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="indoor", choices=["indoor", "outdoor"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ov = {}
    for kv in a.set:
        k, v = kv.split("=")
        try:
            v = float(v) if "." in v or "e" in v.lower() and v.replace("e", "").replace("-", "").replace(".", "").isdigit() else int(v)
        except ValueError:
            pass
        ov[k] = v
    rows = evaluate(a.split, load_config(**ov), a.n, a.workers)
    print(a.split, ov, summarize(rows))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
