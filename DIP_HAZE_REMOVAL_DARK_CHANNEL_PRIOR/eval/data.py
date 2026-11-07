"""Dataset locators. Paths are read from configs/data.yaml, never hard-coded."""
import glob
import os
import cv2
import numpy as np
import yaml

_CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "data.yaml")


def data_cfg():
    with open(_CFG) as f:
        return yaml.safe_load(f)


def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
