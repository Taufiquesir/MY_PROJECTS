"""Temporally regularised DCP: variants V0-V3 (see proposal, Section 2).

V0  per-frame DCP, no temporal handling
V1  EMA on atmospheric light A only
V2  V1 + naive (unwarped) EMA on transmission t
V3  V1 + optical-flow-warped EMA on t, with forward-backward occlusion mask;
    occluded / inconsistent pixels fall back to the raw per-frame t

alpha is the weight on *history*: s_t = alpha * s_{t-1} + (1 - alpha) * x_t.
"""
import time

import numpy as np

from dcp import dehaze, load_config
from dcp.atmosphere import estimate_atmosphere
from dcp.dark_channel import dark_channel
from dcp.recover import recover, depth_from_transmission
from .flow import flow, warp
from .occlusion import valid_mask

VARIANTS = ["V0", "V1", "V2", "V3"]


def dehaze_video(frames, variant="V3", alpha=0.8, cfg=None, return_stages=False, flows=None, A_fixed=None):
    """frames: list of float RGB HxWx3 in [0,1]. Returns dict(J=list, seconds=float, ...).

    flows: optional per-frame (f_cur_to_prev, f_prev_to_cur) override (e.g. ground-truth flow
    for the oracle diagnostic); default estimates flow from the hazy frames themselves.
    A_fixed: optional constant atmospheric light (diagnostic: removes A jitter entirely)."""
    cfg = cfg or load_config()
    t_start = time.perf_counter()
    Js, As, Ts = [], [], []
    A_s = t_s = prev = None
    for k, I in enumerate(frames):
        if variant == "V0":
            out = dehaze(I, cfg, A=A_fixed)
            A_s, t_s, J = out["A"], out["t_ref"], out["J"]
        else:
            # smooth A first, then estimate t *under the smoothed A* so t and A stay consistent
            A_raw = estimate_atmosphere(I, dark_channel(I, cfg["patch_size"]),
                                        cfg["atmosphere"]["estimator"], cfg["atmosphere"]["top_fraction"])
            A_s = A_fixed if A_fixed is not None else (A_raw if A_s is None else alpha * A_s + (1 - alpha) * A_raw)
            out = dehaze(I, cfg, A=A_s)
            t_cur = out["t_ref"]
            if variant == "V1" or t_s is None:
                t_s, J = t_cur, out["J"]
            else:
                if variant == "V2":
                    t_new = alpha * t_s + (1 - alpha) * t_cur
                elif variant == "V3":
                    if flows is not None:
                        f_cp, f_pc = flows[k]
                    else:
                        f_cp = flow(I, prev)                         # cur -> prev
                        f_pc = flow(prev, I)                         # prev -> cur
                    ok = valid_mask(f_cp, f_pc)
                    t_new = np.where(ok, alpha * warp(t_s, f_cp) + (1 - alpha) * t_cur, t_cur)
                else:
                    raise ValueError(variant)
                t_s = t_new
                J = recover(I, A_s, t_s, cfg["t0"])
        prev = I
        Js.append(J)
        if return_stages:
            As.append(np.array(A_s)); Ts.append(t_s)
    res = dict(J=Js, seconds=time.perf_counter() - t_start)
    res["fps"] = len(frames) / res["seconds"]
    if return_stages:
        res.update(A=As, t=Ts)
    return res
