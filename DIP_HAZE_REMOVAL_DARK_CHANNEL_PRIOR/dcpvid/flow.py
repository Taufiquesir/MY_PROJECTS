"""Dense optical flow (OpenCV DIS, classical) and backward warping."""
import cv2
import numpy as np

_dis = None


def _prep(img):
    """Hazy frames are low contrast; stretch to full range before flow estimation."""
    g = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)


def flow(src, dst):
    """Flow f with src(x) ~ dst(x + f(x)); HxWx2 float32 (dx, dy)."""
    global _dis
    if _dis is None:
        _dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _dis.calc(_prep(src), _prep(dst), None)


def warp(img, f):
    """Sample `img` at x + f(x). If f is the flow cur->prev, this warps prev into cur's frame."""
    h, w = f.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img.astype(np.float32), gx + f[..., 0], gy + f[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
