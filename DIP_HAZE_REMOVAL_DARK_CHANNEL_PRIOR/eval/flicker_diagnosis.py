"""Objective 4: which cause of flicker dominates when DCP is applied frame-by-frame?

Controlled interventions on V0 (each removes one candidate cause), scored with the same
flicker index (GT-flow-warped, normalised by the clean GT clip's own temporal change):
  A_frozen     A := per-clip median of per-frame estimates  -> removes atmospheric-light jitter
  t_smoothed   V3 transmission smoothing, A frozen          -> removes transmission jitter (on top)
  t0=0.3/0.05  vary the transmission floor                  -> boundary clipping / noise gain
and a per-t-quartile breakdown of V0's flicker (noise amplification 1/t).
"""
import csv
import os

import numpy as np

from dcp import load_config, dehaze
from dcpvid import dehaze_video
from dcpvid.flow import warp
from eval.revide_runner import load_clip, clips
from eval.temporal_metrics import ClipReference

OUT = "results/flicker_diagnosis.csv"


def per_t_quartile(ref, J, T):
    """flicker (raw, not normalised) of J within quartiles of the (clip-mean-per-frame) t."""
    edges = np.quantile(np.concatenate([t.ravel()[::50] for t in T]), [0.25, 0.5, 0.75])
    num, cnt = np.zeros(4), np.zeros(4)
    for k in range(1, len(J)):
        d = np.abs(J[k] - warp(J[k - 1], ref.f_cp[k])).mean(-1)
        b = np.digitize(T[k], edges)
        for q in range(4):
            m = ref.ok[k] & (b == q)
            num[q] += d[m].sum(); cnt[q] += m.sum()
    return num / np.maximum(cnt, 1)


def main(width=640):
    cfg = load_config()
    rows, quart = [], []
    for clip in clips("Test"):
        hz, gt = load_clip("Test", clip, width)
        ref = ClipReference(gt)
        A_med = np.median([dehaze(h, load_config(**{"refine.method": "none"}))["A"] for h in hz[::3]], axis=0)
        runs = {
            "V0 (baseline)": dict(variant="V0"),
            "A_frozen": dict(variant="V0", A_fixed=A_med),
            "A_frozen + t_smoothed(V3,a=.7)": dict(variant="V3", alpha=0.7, A_fixed=A_med),
            "t0=0.3": dict(variant="V0", cfg=load_config(t0=0.3)),
            "t0=0.05": dict(variant="V0", cfg=load_config(t0=0.05)),
        }
        for name, kw in runs.items():
            kw = dict(kw); v = kw.pop("variant")
            o = dehaze_video(hz, v, kw.pop("alpha", 0.0), kw.pop("cfg", cfg), return_stages=(name == "V0 (baseline)"), **kw)
            rows.append(dict(clip=clip, run=name, flicker_index=ref.flicker_index(o["J"])))
            print(rows[-1], flush=True)
            if name == "V0 (baseline)":
                q = per_t_quartile(ref, o["J"], o["t"])
                quart.append(dict(clip=clip, **{f"t_q{i+1}": q[i] for i in range(4)}, gt_flicker=ref.gt_flicker))
                print(quart[-1], flush=True)
    os.makedirs("results", exist_ok=True)
    for path, data in [(OUT, rows), ("results/flicker_by_t_quartile.csv", quart)]:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)


if __name__ == "__main__":
    main()
