"""Raw transmission from the dark channel of the A-normalised image."""
from .dark_channel import dark_channel


def raw_transmission(img, A, patch_size=15, omega=0.95):
    norm = img / A.reshape(1, 1, 3)
    return 1.0 - omega * dark_channel(norm, patch_size)
