"""Objective 6: sky failure mode on RESIDE SOTS outdoor.

For each image: sky fraction (from the clean GT image, classical detector), PSNR of baseline
DCP and of DCP + sky-aware transmission floor. Writes results/failure_cases.csv and a scatter.
"""
import csv
import os
from multiprocessing import Pool

import numpy as np

from dcp import dehaze, load_config
from dcp.recover import recover
from dcp.sky import sky_mask, sky_floor_transmission
from eval.metrics import psnr, ssim
from eval.reside_runner import list_pairs
from eval.data import read_rgb


def _one(args):
    hp, gp, floor = args
    I, G = read_rgb(hp), read_rgb(gp)
    cfg = load_config()
    out = dehaze(I, cfg)
    t2, _ = sky_floor_transmission(I, out["t_ref"], out["A"], floor=floor)
    J2 = recover(I, out["A"], t2, cfg["t0"])
    return dict(image=os.path.basename(hp), sky_frac=float(sky_mask(G).mean()),
                psnr_base=psnr(out["J"], G), psnr_mit=psnr(J2, G),
                ssim_base=ssim(out["J"], G), ssim_mit=ssim(J2, G))


def main(n=200, floor=0.6, workers=4, out="results/failure_cases.csv"):
    jobs = [(h, g, floor) for h, g in list_pairs("outdoor", n)]
    with Pool(workers) as p:
        rows = p.map(_one, jobs, chunksize=2)
    os.makedirs("results", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sf = np.array([r["sky_frac"] for r in rows])
    for name, m in [("all", sf >= 0), ("sky<5%", sf < 0.05), ("sky 5-25%", (sf >= .05) & (sf < .25)), ("sky>=25%", sf >= .25)]:
        if m.any():
            print(f"{name:10s} n={m.sum():3d} base PSNR {np.mean([r['psnr_base'] for r, k in zip(rows, m) if k]):.2f}"
                  f"  mitigated {np.mean([r['psnr_mit'] for r, k in zip(rows, m) if k]):.2f}")
    print("corr(sky_frac, PSNR_base) =", round(float(np.corrcoef(sf, [r["psnr_base"] for r in rows])[0, 1]), 3))


if __name__ == "__main__":
    main()
