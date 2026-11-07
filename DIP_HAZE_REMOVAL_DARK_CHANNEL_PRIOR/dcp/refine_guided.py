"""Guided filter (He, Sun, Tang, ECCV 2010) with O(N) box filters, plus fast variant."""
import cv2
import numpy as np


def _box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter(I, p, r, eps):
    """I: HxW or HxWx3 guide in [0,1]; p: HxW input. Returns filtered p (HxW)."""
    if I.ndim == 2:
        I = I[..., None]
    h, w, c = I.shape
    mean_I = np.stack([_box(I[..., k], r) for k in range(c)], -1)
    mean_p = _box(p, r)
    mean_Ip = np.stack([_box(I[..., k] * p, r) for k in range(c)], -1)
    cov_Ip = mean_Ip - mean_I * mean_p[..., None]
    # covariance matrix of I per pixel (c x c)
    var = np.empty((h, w, c, c), dtype=np.float64)
    for i in range(c):
        for j in range(i, c):
            v = _box(I[..., i] * I[..., j], r) - mean_I[..., i] * mean_I[..., j]
            var[..., i, j] = v
            var[..., j, i] = v
    var += eps * np.eye(c)
    a = np.linalg.solve(var, cov_Ip[..., None])[..., 0]            # HxWxc
    b = mean_p - (a * mean_I).sum(-1)
    mean_a = np.stack([_box(a[..., k], r) for k in range(c)], -1)
    mean_b = _box(b, r)
    return (mean_a * I).sum(-1) + mean_b


def fast_guided_filter(I, p, r, eps, s=4):
    """Fast guided filter: compute a,b on s-subsampled images (radius r/s),
    upsample a,b, apply at full resolution. Cost O(N/s^2) for the heavy part."""
    h, w = p.shape
    if I.ndim == 2:
        I = I[..., None]
    c = I.shape[2]
    hs, ws = max(1, h // s), max(1, w // s)
    Is = cv2.resize(I if c > 1 else I[..., 0], (ws, hs), interpolation=cv2.INTER_NEAREST)
    if Is.ndim == 2:
        Is = Is[..., None]
    ps = cv2.resize(p, (ws, hs), interpolation=cv2.INTER_NEAREST)
    rs = max(1, r // s)
    mean_I = np.stack([_box(Is[..., k], rs) for k in range(c)], -1)
    mean_p = _box(ps, rs)
    cov_Ip = np.stack([_box(Is[..., k] * ps, rs) for k in range(c)], -1) - mean_I * mean_p[..., None]
    var = np.empty((hs, ws, c, c))
    for i in range(c):
        for j in range(i, c):
            v = _box(Is[..., i] * Is[..., j], rs) - mean_I[..., i] * mean_I[..., j]
            var[..., i, j] = v
            var[..., j, i] = v
    var += eps * np.eye(c)
    a = np.linalg.solve(var, cov_Ip[..., None])[..., 0]
    b = mean_p - (a * mean_I).sum(-1)
    mean_a = np.stack([_box(a[..., k], rs) for k in range(c)], -1)
    mean_b = _box(b, rs)
    a_up = np.stack([cv2.resize(mean_a[..., k], (w, h), interpolation=cv2.INTER_LINEAR) for k in range(c)], -1)
    b_up = cv2.resize(mean_b, (w, h), interpolation=cv2.INTER_LINEAR)
    return (a_up * I).sum(-1) + b_up
