"""End-to-end single-image DCP. Every stage's output is returned for figures/debugging."""
import copy
import os
import yaml
import numpy as np

from .dark_channel import dark_channel
from .atmosphere import estimate_atmosphere
from .transmission import raw_transmission
from .refine_guided import guided_filter, fast_guided_filter
from .refine_matting import soft_matting
from .recover import recover, depth_from_transmission

_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


def load_config(path=None, **overrides):
    with open(path or _DEFAULT) as f:
        cfg = yaml.safe_load(f)
    for k, v in overrides.items():          # dotted keys: refine.method='matting'
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = v
    return cfg


def refine_transmission(img, t_raw, rc):
    m = rc["method"]
    if m == "none":
        return t_raw
    if m == "guided":
        gray = img.mean(axis=2)
        return np.clip(guided_filter(gray, t_raw, rc["guided_radius"], rc["guided_eps"]), 0, 1)
    if m == "fast_guided":
        gray = img.mean(axis=2)
        return np.clip(fast_guided_filter(gray, t_raw, rc["guided_radius"], rc["guided_eps"], rc["fast_subsample"]), 0, 1)
    if m == "matting":
        return soft_matting(img, t_raw, rc["matting_eps"], rc["matting_lambda"], rc["matting_radius"])
    raise ValueError(f"unknown refine method: {m}")


def dehaze(img, cfg=None, A=None):
    """img: float64 HxWx3 RGB in [0,1]. If A is given it overrides the estimator
    (used by the video extension to inject a temporally smoothed A)."""
    cfg = cfg or load_config()
    img = np.asarray(img, dtype=np.float64)
    dark = dark_channel(img, cfg["patch_size"])
    if A is None:
        A = estimate_atmosphere(img, dark, cfg["atmosphere"]["estimator"], cfg["atmosphere"]["top_fraction"])
    t_raw = raw_transmission(img, A, cfg["patch_size"], cfg["omega"])
    t_ref = refine_transmission(img, t_raw, cfg["refine"])
    J = recover(img, A, t_ref, cfg["t0"])
    return dict(dark=dark, A=A, t_raw=t_raw, t_ref=t_ref, J=J, depth=depth_from_transmission(t_ref))
