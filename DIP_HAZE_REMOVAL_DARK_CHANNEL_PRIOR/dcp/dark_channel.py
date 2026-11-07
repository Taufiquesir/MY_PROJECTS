"""Dark channel: per-pixel min over colour channels, then a patch-wise min (erosion)."""
import cv2
import numpy as np


def dark_channel(img, patch_size=15):
    """img: float HxWx3 in [0,1]. Returns HxW dark channel."""
    mn = img.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(mn, kernel)
