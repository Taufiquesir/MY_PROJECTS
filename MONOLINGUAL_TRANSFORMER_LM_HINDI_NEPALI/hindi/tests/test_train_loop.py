"""End-to-end tests for the training loop, on tiny synthetic data and a tiny model
so they run fast -- these are not meant to test that the model learns anything
(random synthetic tokens carry no signal), only that the training/resume/checkpoint
machinery itself is correct.
"""

import numpy as np
import pytest

from hindi.model.checkpoint import load_checkpoint
from hindi.model.config import ModelConfig
from hindi.model.train_loop import TrainConfig, train


@pytest.fixture
def tiny_bins(tmp_path):
    rng = np.random.default_rng(0)
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    rng.integers(0, 50, size=5000, dtype=np.uint16).tofile(train_path)
    rng.integers(0, 50, size=1000, dtype=np.uint16).tofile(val_path)
    return train_path, val_path


def _tiny_model_cfg() -> ModelConfig:
    return ModelConfig(vocab_size=50, d_model=16, n_layers=2, n_heads=2, d_ff=32, max_seq_len=8, dropout=0.0)


def test_training_runs_and_checkpoints(tmp_path, tiny_bins):
    train_bin, val_bin = tiny_bins
    train_cfg = TrainConfig(
        max_steps=6, warmup_steps=1, batch_size=4, eval_every=3, eval_batches=2, checkpoint_every=3, log_every=100
    )
    ckpt_dir = tmp_path / "ckpt"
    train(_tiny_model_cfg(), train_cfg, train_bin, val_bin, ckpt_dir, device="cpu", resume=False)

    assert (ckpt_dir / "latest.pt").exists()
    assert (ckpt_dir / "best.pt").exists()

    _, _, _, state = load_checkpoint(ckpt_dir / "latest.pt")
    assert state.step == 6


def test_resume_continues_from_correct_step(tmp_path, tiny_bins):
    train_bin, val_bin = tiny_bins
    model_cfg = _tiny_model_cfg()
    ckpt_dir = tmp_path / "ckpt"

    partial_cfg = TrainConfig(
        max_steps=3, warmup_steps=1, batch_size=4, eval_every=3, eval_batches=2, checkpoint_every=3, log_every=100
    )
    train(model_cfg, partial_cfg, train_bin, val_bin, ckpt_dir, device="cpu", resume=False)
    _, _, _, state_after_partial = load_checkpoint(ckpt_dir / "latest.pt")
    assert state_after_partial.step == 3

    full_cfg = TrainConfig(
        max_steps=6, warmup_steps=1, batch_size=4, eval_every=3, eval_batches=2, checkpoint_every=3, log_every=100
    )
    train(model_cfg, full_cfg, train_bin, val_bin, ckpt_dir, device="cpu", resume=True)
    _, _, _, state_after_resume = load_checkpoint(ckpt_dir / "latest.pt")
    assert state_after_resume.step == 6
