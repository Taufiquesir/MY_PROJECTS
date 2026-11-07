"""Per-stage unit tests with hand-checkable expectations."""
import numpy as np
import pytest

from dcp.atmosphere import estimate_atmosphere
from dcp.dark_channel import dark_channel
from dcp.recover import recover
from dcp.refine_guided import guided_filter, fast_guided_filter
from dcp.refine_matting import matting_laplacian
from dcp.sky import sky_mask
from dcp.transmission import raw_transmission
from dcpvid.flow import warp
from dcpvid.occlusion import valid_mask
from eval.metrics import psnr, ssim, ciede2000


def test_dark_channel_min_over_channels_and_patch():
    img = np.full((9, 9, 3), 0.8)
    img[4, 4] = [0.8, 0.1, 0.9]                      # one dark pixel in G
    d = dark_channel(img, 3)
    assert d[4, 4] == pytest.approx(0.1) and d[3, 3] == pytest.approx(0.1)   # inside 3x3 of centre
    assert d[0, 0] == pytest.approx(0.8)                                     # far corner untouched


def test_dark_channel_is_le_per_pixel_min():
    rng = np.random.default_rng(1)
    img = rng.random((20, 20, 3))
    assert (dark_channel(img, 5) <= img.min(2) + 1e-12).all()


def test_atmosphere_brightest_and_mean():
    img = np.zeros((20, 20, 3)); dark = np.zeros((20, 20))
    img[0, 0] = [0.9, 0.9, 0.9]; img[0, 1] = [0.5, 0.5, 0.5]
    dark[0, 0] = dark[0, 1] = 1.0                    # both are top-dark-channel candidates
    A = estimate_atmosphere(img, dark, "brightest", top_fraction=0.005)   # 2 of 400 pixels
    assert np.allclose(A, 0.9)
    assert np.allclose(estimate_atmosphere(img, dark, "mean", top_fraction=0.005), 0.7)
    with pytest.raises(ValueError):
        estimate_atmosphere(img, dark, "nope")


def test_raw_transmission_of_haze_free_dark_image_is_near_one():
    img = np.zeros((16, 16, 3)); img[..., 0] = 0.5          # G,B = 0 -> dark channel 0
    t = raw_transmission(img, np.array([0.9, 0.9, 0.9]), 5, 0.95)
    assert np.allclose(t, 1.0)


def test_recover_inverts_scattering_model():
    rng = np.random.default_rng(2)
    J = rng.random((8, 8, 3)) * 0.8; t = np.full((8, 8), 0.5); A = np.array([0.9, 0.8, 0.7])
    I = J * t[..., None] + A * (1 - t[..., None])
    assert np.allclose(recover(I, A, t, t0=0.1), J, atol=1e-9)


def test_recover_floors_transmission():
    I = np.full((2, 2, 3), 0.2); A = np.ones(3)
    out = recover(I, A, np.zeros((2, 2)), t0=0.5)           # (0.2-1)/0.5+1 = -0.6 -> clipped to 0
    assert (out == 0).all()


def test_guided_filter_identity_and_constant():
    rng = np.random.default_rng(3)
    g = rng.random((30, 30))
    out = guided_filter(g, g, r=3, eps=1e-8)                # guide == input -> ~identity (tiny eps)
    assert np.abs(out - g).mean() < 0.05
    assert np.allclose(guided_filter(g, np.full((30, 30), 0.4), 3, 1e-3), 0.4, atol=1e-6)


def test_fast_guided_close_to_guided():
    rng = np.random.default_rng(4)
    g = cv_smooth(rng.random((64, 64)))
    p = g + 0.05 * rng.standard_normal(g.shape)
    a, b = guided_filter(g, p, 8, 1e-3), fast_guided_filter(g, p, 8, 1e-3, s=2)
    assert np.abs(a - b).mean() < 0.03


def cv_smooth(x):
    import cv2
    return cv2.GaussianBlur(x, (0, 0), 3)


def test_matting_laplacian_properties():
    rng = np.random.default_rng(5)
    img = rng.random((10, 10, 3))
    L = matting_laplacian(img, eps=1e-5, r=1)
    assert abs(L - L.T).max() < 1e-9                         # symmetric
    assert np.abs(L @ np.ones(100)).max() < 1e-6              # constants are in the null space


def test_sky_mask_needs_top_connection():
    img = np.full((20, 20, 3), 0.3)
    img[0:5] = 0.9                                           # bright, grey, touches top
    img[12:16, 4:8] = 0.9                                    # bright but isolated -> not sky
    m = sky_mask(img)
    assert m[2, 5] and not m[13, 5]


def test_warp_integer_shift_and_validity_of_zero_flow():
    img = np.arange(100, dtype=np.float32).reshape(10, 10)
    f = np.zeros((10, 10, 2), np.float32); f[..., 0] = 2     # sample at x+2
    w = warp(img, f)
    assert np.allclose(w[:, :7], img[:, 2:9])
    z = np.zeros((10, 10, 2), np.float32)
    assert valid_mask(z, z).all()


def test_valid_mask_flags_inconsistent_flow():
    fcp = np.zeros((20, 20, 2), np.float32); fcp[..., 0] = 5
    fpc = np.zeros((20, 20, 2), np.float32); fpc[..., 0] = 5     # should be -5 to be consistent
    assert not valid_mask(fcp, fpc)[10, 5]


def test_metrics_identity_and_ordering():
    rng = np.random.default_rng(6)
    a = rng.random((32, 32, 3)); b = np.clip(a + 0.1 * rng.standard_normal(a.shape), 0, 1)
    assert ssim(a, a) == pytest.approx(1.0) and ciede2000(a, a) == pytest.approx(0.0)
    assert psnr(a, a) > 100 and psnr(a, b) < 40


def test_soft_matting_converges_and_stays_close_to_raw_t():
    """Regression: CG must reach its tolerance (an iteration cap that was too low gave unconverged,
    run-to-run varying results at full size)."""
    import warnings
    from dcp.refine_matting import soft_matting
    rng = np.random.default_rng(7)
    img = cv_smooth(rng.random((48, 64))); img = np.stack([img, img * 0.9, img * 0.8], -1)
    t_raw = np.clip(0.5 + 0.1 * rng.standard_normal((48, 64)), 0, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # a non-convergence warning fails the test
        t = soft_matting(img, t_raw)
    assert t.shape == t_raw.shape and 0 <= t.min() and t.max() <= 1
    assert np.abs(t - t_raw).mean() < 0.2
