"""Synthetic haze generation with known ground truth (A, t)."""
import cv2
import numpy as np


def make_scene(h=240, w=320, seed=0):
    """Haze-free scene satisfying the dark-channel prior: saturated colour blobs
    where one channel is near zero (as in foliage, shadows, coloured objects)."""
    rng = np.random.default_rng(seed)
    J = np.zeros((h, w, 3))
    for _ in range(60):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        ry, rx = rng.integers(10, 50), rng.integers(10, 50)
        col = rng.random(3) * 0.6
        col[rng.integers(0, 3)] = rng.random() * 0.03       # one near-zero channel
        mask = np.zeros((h, w), np.float32)
        cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 1.0, -1)
        J = np.where(mask[..., None] > 0, col, J)
    return np.clip(J + 0.01 * rng.standard_normal(J.shape), 0, 1)


def make_depth(h=240, w=320):
    """Smooth depth increasing towards the top of the image (horizon)."""
    y = np.linspace(1.0, 0.0, h)[:, None] * np.ones((1, w))
    return 0.2 + 2.8 * y


def add_haze(J, depth, A, beta=1.0):
    t = np.exp(-beta * depth)
    I = J * t[..., None] + A.reshape(1, 1, 3) * (1 - t[..., None])
    return I, t
