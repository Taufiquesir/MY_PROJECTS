"""Objective 1 / Deliverable 2: round-trip test against known ground truth."""
import numpy as np
from dcp import dehaze, load_config
from eval.synthetic import make_scene, make_depth, add_haze

A_TRUE = np.array([0.85, 0.88, 0.92])


def _setup():
    J = make_scene()
    I, t = add_haze(J, make_depth(), A_TRUE, beta=1.0)
    return J, I, t


def test_atmospheric_light_recovered():
    _, I, _ = _setup()
    out = dehaze(I, load_config(**{"refine.method": "none"}))
    assert np.abs(out["A"] - A_TRUE).max() < 0.05


def test_transmission_recovered_guided():
    _, I, t = _setup()
    out = dehaze(I, load_config(**{"refine.method": "guided"}))
    assert np.abs(out["t_ref"] - t).mean() < 0.08


def test_radiance_improves_over_hazy():
    J, I, _ = _setup()
    out = dehaze(I, load_config(**{"refine.method": "guided"}))
    assert np.abs(out["J"] - J).mean() < 0.5 * np.abs(I - J).mean()


def test_all_refiners_run_and_stay_bounded():
    J, I, t = _setup()
    small = (slice(0, 80), slice(0, 100))
    for m in ["none", "guided", "fast_guided", "matting"]:
        out = dehaze(I[small], load_config(**{"refine.method": m}))
        assert out["t_ref"].min() >= 0 and out["t_ref"].max() <= 1
        assert out["J"].shape == I[small].shape
