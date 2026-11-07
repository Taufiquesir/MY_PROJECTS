"""Classical (non-learned) sky detection and a sky-aware transmission floor (mitigation)."""
import cv2
import numpy as np


def sky_mask(img, bright=0.7, sat=0.20, min_frac=0.0):
    """Sky = bright, low-saturation pixels in connected components touching the top image row."""
    mx, mn = img.max(2), img.min(2)
    cand = ((mx > bright) & ((mx - mn) < sat)).astype(np.uint8)
    n, lab = cv2.connectedComponents(cand, connectivity=4)
    top = np.unique(lab[0][lab[0] > 0])
    return np.isin(lab, top)


def sky_floor_transmission(img, t, A, floor=0.6, feather=15):
    """Raise t to `floor` inside the (feathered) detected sky so J there stays near the input
    instead of being amplified by 1/t. Sky detected on the hazy image: pixels within 10% of A,
    low saturation, connected to the top row."""
    mx, mn = img.max(2), img.min(2)
    near_A = (np.abs(img - A.reshape(1, 1, 3)).max(2) < 0.10) & ((mx - mn) < 0.20)
    n, lab = cv2.connectedComponents(near_A.astype(np.uint8), connectivity=4)
    top = np.unique(lab[0][lab[0] > 0])
    m = np.isin(lab, top).astype(np.float32)
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather / 3)
    return np.maximum(t, floor * m + t * (1 - m)) if floor else t, m
