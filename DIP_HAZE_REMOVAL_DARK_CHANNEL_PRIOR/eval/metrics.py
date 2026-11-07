"""Image-fidelity metrics. Inputs: float RGB HxWx3 in [0,1]."""
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(a, b):
    return float(peak_signal_noise_ratio(b, a, data_range=1.0))


def ssim(a, b):
    return float(structural_similarity(b, a, channel_axis=2, data_range=1.0))


def ciede2000(a, b):
    """Mean CIEDE2000 colour difference."""
    return float(deltaE_ciede2000(rgb2lab(b), rgb2lab(a)).mean())
