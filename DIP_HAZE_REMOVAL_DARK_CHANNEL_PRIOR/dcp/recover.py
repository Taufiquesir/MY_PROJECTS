"""Scene radiance recovery: J = (I - A) / max(t, t0) + A."""
import numpy as np


def recover(img, A, t, t0=0.1):
    t = np.maximum(t, t0)[..., None]
    return np.clip((img - A.reshape(1, 1, 3)) / t + A.reshape(1, 1, 3), 0.0, 1.0)


def depth_from_transmission(t, beta=1.0):
    """Relative depth d = -ln(t)/beta (visualisation only)."""
    import numpy as np
    return -np.log(np.clip(t, 1e-3, 1.0)) / beta
