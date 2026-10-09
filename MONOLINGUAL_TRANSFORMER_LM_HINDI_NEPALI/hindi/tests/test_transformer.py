"""Unit tests for the hand-implemented decoder-only Transformer."""

import torch
import torch.nn.functional as F

from hindi.model.config import ModelConfig
from hindi.model.transformer import DecoderOnlyTransformer


def _tiny_model(**overrides) -> DecoderOnlyTransformer:
    cfg = ModelConfig(
        vocab_size=100, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=16, dropout=0.0
    )
    cfg.__dict__.update(overrides)
    return DecoderOnlyTransformer(**cfg.__dict__)


def test_forward_pass_output_shape():
    model = _tiny_model()
    B, T = 2, 10
    x = torch.randint(0, 100, (B, T))
    logits = model(x)
    assert logits.shape == (B, T, 100)


def test_causal_mask_blocks_future_tokens():
    """Changing the last input token must not change logits at earlier positions --
    the assignment's own required empirical check that the model cannot see the future.
    """
    torch.manual_seed(0)
    model = _tiny_model()
    model.eval()

    x = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        logits1 = model(x)

    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % 100
    with torch.no_grad():
        logits2 = model(x2)

    early_diff = (logits1[:, :-1, :] - logits2[:, :-1, :]).abs().max().item()
    last_diff = (logits1[:, -1, :] - logits2[:, -1, :]).abs().max().item()
    assert early_diff == 0.0, "future token leaked into earlier-position logits"
    assert last_diff > 0.0, "changing the input had no effect anywhere (suspicious)"


def test_weight_tying():
    model = _tiny_model()
    assert model.lm_head.weight is model.token_embedding.weight


def test_gradients_flow_to_every_parameter():
    model = _tiny_model()
    x = torch.randint(0, 100, (2, 8))
    logits = model(x)
    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, 100), x[:, 1:].reshape(-1))
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"


def test_sequence_longer_than_max_seq_len_raises():
    model = _tiny_model()
    x = torch.randint(0, 100, (1, 32))  # max_seq_len is 16
    try:
        model(x)
        assert False, "expected ValueError for sequence longer than max_seq_len"
    except ValueError:
        pass


def test_param_count_meets_25m_floor_for_shipped_config():
    """The actual per-language config (vocab_size=9000) must be at least 25M
    trainable parameters -- a hard floor, not just "close to" the assignment's
    ~25M target -- while staying close enough to 25M to still be justifiable as
    that target (not, say, 2x over)."""
    cfg = ModelConfig(vocab_size=9000)
    model = DecoderOnlyTransformer(**cfg.__dict__)
    total = model.num_parameters()
    assert total >= 25_000_000, f"param count {total:,} is under the 25M floor"
    assert total <= 28_000_000, f"param count {total:,} is too far above the ~25M target"
