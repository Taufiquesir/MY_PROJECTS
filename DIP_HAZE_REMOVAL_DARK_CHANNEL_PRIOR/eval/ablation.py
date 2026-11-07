"""One-factor-at-a-time ablation on RESIDE SOTS (Deliverable 3).

Baseline = configs/default.yaml. Each row varies exactly one factor.
The refinement-method comparison is run on the *same* fixed subset for all methods
(matting is ~40 s/image); the other factors run on the full test split.
"""
import csv
import os
import sys

from dcp import load_config
from eval.reside_runner import evaluate, summarize

FACTORS = {
    "patch_size": [7, 15, 31],
    "omega": [0.80, 0.90, 0.95, 1.00],
    "atmosphere.estimator": ["brightest", "mean"],
    "t0": [0.05, 0.1, 0.2, 0.3],
}
METHODS = ["none", "guided", "fast_guided", "matting"]
MATTING_N = 25


def _key(split, group, factor, value):
    return (split, group, factor, str(value))


def main(out="results/single_image_ablation.csv", workers=8):
    """Resumable: every finished configuration is appended to `out` immediately and
    skipped on re-run (long runs get interrupted)."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        with open(out) as fh:
            done = {_key(r["split"], r["group"], r["factor"], r["value"]) for r in csv.DictReader(fh)}
    fields = ["split", "group", "factor", "value", "psnr", "ssim", "ciede2000", "seconds", "n"]

    def record(split, group, factor, value, n, cfg):
        if _key(split, group, factor, value) in done:
            return
        w = 2 if cfg["refine"]["method"] == "matting" else workers   # sparse Laplacian ~GBs/image; 7 GB RAM box
        s = summarize(evaluate(split, cfg, n, w))
        row = dict(split=split, group=group, factor=factor, value=value, **s)
        new = not os.path.exists(out)
        with open(out, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            if new:
                wr.writeheader()
            wr.writerow(row)
        print(row, flush=True)

    for split in ["indoor", "outdoor"]:
        for m in METHODS:                                       # refinement: shared subset
            record(split, "refine_subset", "refine.method", m, MATTING_N, load_config(**{"refine.method": m}))
        for f, vals in FACTORS.items():                         # OFAT on full split
            for v in vals:
                record(split, "ofat_full", f, v, None, load_config(**{f: v}))
        for m in ["none", "guided", "fast_guided"]:             # refinement on full split (no matting)
            record(split, "refine_full", "refine.method", m, None, load_config(**{"refine.method": m}))


if __name__ == "__main__":
    main(*sys.argv[1:2])
