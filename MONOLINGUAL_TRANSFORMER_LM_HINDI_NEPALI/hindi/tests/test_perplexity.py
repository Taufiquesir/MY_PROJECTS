"""Unit tests for eval/perplexity.py -- verified end-to-end against a tiny model
and a synthetic token stream, since no real checkpoint exists yet at
test-authoring time.
"""

import math
from pathlib import Path

import numpy as np
import torch

from hindi.eval.perplexity import evaluate_perplexity
from hindi.model.checkpoint import save_checkpoint, TrainingState
from hindi.model.config import ModelConfig
from hindi.model.train_loop import build_lr_scheduler, build_optimizer, TrainConfig
from hindi.model.transformer import DecoderOnlyTransformer


def _tiny_checkpoint(tmp_path: Path) -> Path:
    model_cfg = ModelConfig(vocab_size=50, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=8, dropout=0.0)
    train_cfg = TrainConfig(lr=1e-3, max_steps=10, warmup_steps=1)
    model = DecoderOnlyTransformer(**model_cfg.__dict__)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_lr_scheduler(optimizer, train_cfg)
    ckpt_path = tmp_path / "test.pt"
    save_checkpoint(ckpt_path, model, optimizer, scheduler, model_cfg, TrainingState(step=0, best_val_loss=float("inf")))
    return ckpt_path


def _synthetic_bin(tmp_path: Path, n_tokens: int, vocab_size: int) -> Path:
    ids = np.random.randint(0, vocab_size, size=n_tokens).astype(np.uint16)
    path = tmp_path / "eval.bin"
    ids.tofile(path)
    return path


def test_ppl_is_exp_of_mean_nll(tmp_path):
    ckpt_path = _tiny_checkpoint(tmp_path)
    bin_path = _synthetic_bin(tmp_path, n_tokens=200, vocab_size=50)

    result = evaluate_perplexity(ckpt_path, bin_path, bytes_per_token=4.0, device="cpu", batch_size=4)

    assert math.isclose(result.ppl, math.exp(result.mean_nll), rel_tol=1e-6)


def test_bpb_scales_inversely_with_bytes_per_token(tmp_path):
    """Same mean_nll, but a larger bytes_per_token ratio must produce a smaller
    BPB (more bytes to spread the same number of bits over -> fewer bits/byte)."""
    ckpt_path = _tiny_checkpoint(tmp_path)
    bin_path = _synthetic_bin(tmp_path, n_tokens=200, vocab_size=50)

    result_low = evaluate_perplexity(ckpt_path, bin_path, bytes_per_token=2.0, device="cpu", batch_size=4)
    result_high = evaluate_perplexity(ckpt_path, bin_path, bytes_per_token=8.0, device="cpu", batch_size=4)

    # mean_nll is identical (same model, same data) -- only bytes_per_token differs.
    assert math.isclose(result_low.mean_nll, result_high.mean_nll, rel_tol=1e-6)
    assert math.isclose(result_low.bpb, result_high.bpb * 4, rel_tol=1e-6)


def test_ppl_of_uniform_random_model_is_near_vocab_size(tmp_path):
    """An untrained model with small random-init weights produces near-uniform
    next-token logits, so its PPL should be close to (not exactly, since init isn't
    perfectly uniform) the vocab size -- the same sanity check used during real
    training (ln(vocab_size) as the expected starting loss)."""
    vocab_size = 50
    ckpt_path = _tiny_checkpoint(tmp_path)
    bin_path = _synthetic_bin(tmp_path, n_tokens=400, vocab_size=vocab_size)

    result = evaluate_perplexity(ckpt_path, bin_path, bytes_per_token=4.0, device="cpu", batch_size=8)

    # Loose bound: within an order of magnitude of vocab_size, not exact equality --
    # small-init weights aren't perfectly uniform, but should be in the right regime.
    assert vocab_size / 5 < result.ppl < vocab_size * 5


def test_n_tokens_evaluated_matches_dataset_size(tmp_path):
    ckpt_path = _tiny_checkpoint(tmp_path)
    n_tokens = 200
    bin_path = _synthetic_bin(tmp_path, n_tokens=n_tokens, vocab_size=50)

    result = evaluate_perplexity(ckpt_path, bin_path, bytes_per_token=4.0, device="cpu", batch_size=4)

    # TokenWindowDataset yields (n_tokens - seq_len) windows; batch_size=4 with
    # drop_last=True means the count evaluated is a multiple of 4*seq_len, not
    # necessarily every possible window -- just check it's positive and bounded by
    # the total possible.
    max_seq_len = 8
    assert 0 < result.n_tokens_evaluated <= (n_tokens - max_seq_len) * max_seq_len
