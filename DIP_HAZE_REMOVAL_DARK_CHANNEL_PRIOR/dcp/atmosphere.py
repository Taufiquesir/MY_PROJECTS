"""Atmospheric light estimation (He et al., Sec. 4.4)."""
import numpy as np


def estimate_atmosphere(img, dark, estimator="brightest", top_fraction=0.001):
    """Pick the top `top_fraction` pixels by dark-channel value, then among
    them choose the pixel(s) of highest input intensity.

    estimator='brightest': single brightest pixel (sum over channels) -> A (3,)
    estimator='mean'     : mean colour of all candidate pixels -> A (3,)
    """
    h, w = dark.shape
    n = max(1, int(h * w * top_fraction))
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    cand = img.reshape(-1, 3)[idx]
    if estimator == "brightest":
        return cand[np.argmax(cand.sum(axis=1))].astype(np.float64)
    if estimator == "mean":
        return cand.mean(axis=0).astype(np.float64)
    raise ValueError(f"unknown atmosphere estimator: {estimator}")
