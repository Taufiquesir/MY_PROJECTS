"""Unit tests for the bonus no-positional-embedding ablation model.

Mirrors hindi/tests/test_transformer.py's structure for the real model -- the
causal-mask test in particular is the same empirical check the assignment
requires for the real model, run again here to confirm removing RoPE didn't also
silently break causal masking (a separate property from positional information).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hindi.model.config import ModelConfig
from hindi.model.transformer import DecoderOnlyTransformer
from bonus.model.transformer_ablation import DecoderOnlyTransformerNoPos


def _tiny_ablation_model(**overrides) -> DecoderOnlyTransformerNoPos:
    cfg = ModelConfig(
        vocab_size=100, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=16, dropout=0.0
    )
    cfg.__dict__.update(overrides)
    return DecoderOnlyTransformerNoPos(**cfg.__dict__)


def test_forward_pass_output_shape():
    model = _tiny_ablation_model()
    B, T = 2, 10
    x = torch.randint(0, 100, (B, T))
    logits = model(x)
    assert logits.shape == (B, T, 100)


def test_causal_mask_still_blocks_future_tokens():
    """Removing RoPE must not accidentally also break causal masking -- these are
    two independent properties (mask is applied to attention scores regardless of
    whether Q/K were rotated first)."""
    torch.manual_seed(0)
    model = _tiny_ablation_model()
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


def test_no_rope_buffers_registered():
    """The whole ablation is that no rotation ever happens -- confirm the model
    genuinely has no rope_cos/rope_sin buffers anywhere, not just that they're
    unused."""
    model = _tiny_ablation_model()
    for name, _ in model.named_buffers():
        assert "rope" not in name.lower(), f"found a RoPE-related buffer: {name}"


def test_same_parameter_count_as_real_model():
    """The ablation must be a fair comparison -- same architecture size, only the
    positional scheme differs, so the two models should have identical parameter
    counts (RoPE itself adds zero learnable parameters, so removing it changes
    nothing about the count)."""
    cfg = ModelConfig(vocab_size=100, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=16, dropout=0.0)
    real_model = DecoderOnlyTransformer(**cfg.__dict__)
    ablation_model = DecoderOnlyTransformerNoPos(**cfg.__dict__)
    assert ablation_model.num_parameters() == real_model.num_parameters()


def test_weight_tying():
    model = _tiny_ablation_model()
    assert model.lm_head.weight is model.token_embedding.weight
