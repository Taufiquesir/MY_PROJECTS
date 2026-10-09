"""Unit tests for eval/attention_analysis.py -- verified against a tiny model since
no real checkpoint exists yet at test-authoring time.
"""

import torch

from hindi.eval.attention_analysis import (
    attention_entropy,
    capture_attention,
    mean_attention_distance,
    summarize_attention,
)
from hindi.model.config import ModelConfig
from hindi.model.transformer import DecoderOnlyTransformer


def _tiny_model(**overrides) -> DecoderOnlyTransformer:
    cfg = ModelConfig(
        vocab_size=100, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=16, dropout=0.0
    )
    cfg.__dict__.update(overrides)
    return DecoderOnlyTransformer(**cfg.__dict__)


class _FakeTokenizer:
    def id_to_piece(self, i: int) -> str:
        return f"<{i}>"


def test_capture_attention_shapes_match_model_config():
    model = _tiny_model()
    T = 6
    input_ids = torch.randint(0, 100, (1, T))
    capture = capture_attention(model, input_ids, _FakeTokenizer(), device="cpu")

    assert len(capture.weights) == 2  # n_layers
    for layer_weights in capture.weights:
        assert layer_weights.shape == (4, T, T)  # (n_heads, T, T)
    assert len(capture.tokens) == T


def test_captured_attention_rows_sum_to_one():
    """Every query position's attention distribution over unmasked keys must be a
    valid probability distribution (softmax output), same invariant the model's own
    forward pass relies on."""
    model = _tiny_model()
    T = 8
    input_ids = torch.randint(0, 100, (1, T))
    capture = capture_attention(model, input_ids, _FakeTokenizer(), device="cpu")

    for layer_weights in capture.weights:
        row_sums = layer_weights.sum(dim=-1)  # (h, T)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_captured_attention_respects_causal_mask():
    """A query position must assign exactly zero probability to any future key
    position -- the same causality property tested directly on the model's forward
    pass in test_transformer.py, verified here for the recomputed weights too."""
    model = _tiny_model()
    T = 8
    input_ids = torch.randint(0, 100, (1, T))
    capture = capture_attention(model, input_ids, _FakeTokenizer(), device="cpu")

    for layer_weights in capture.weights:
        for i in range(T):
            future_mass = layer_weights[:, i, i + 1 :].sum()
            assert future_mass.item() == 0.0


def test_attention_entropy_uniform_distribution_is_log_n():
    """A uniform attention distribution over n unmasked keys has entropy log(n) --
    the maximum possible entropy for n outcomes, a known closed-form sanity check."""
    n = 4
    uniform = torch.full((1, 1, n), 1.0 / n)  # (h=1, T=1, T=n)
    entropy = attention_entropy(uniform)
    assert torch.allclose(entropy, torch.tensor([[float(torch.log(torch.tensor(n * 1.0)))]]), atol=1e-5)


def test_attention_entropy_one_hot_distribution_is_zero():
    """A distribution that puts all mass on one key has zero entropy -- the other
    extreme from the uniform case above."""
    one_hot = torch.zeros(1, 1, 4)
    one_hot[0, 0, 2] = 1.0
    entropy = attention_entropy(one_hot)
    assert torch.allclose(entropy, torch.zeros(1, 1), atol=1e-5)


def test_mean_attention_distance_all_self_attention_is_zero():
    """If every query position attends entirely to itself (identity attention),
    mean distance must be exactly 0 (|i - i| = 0 for every position)."""
    T = 5
    weights = torch.eye(T).unsqueeze(0)  # (h=1, T, T), identity = attend only to self
    distance = mean_attention_distance(weights)
    assert torch.allclose(distance, torch.zeros(1), atol=1e-5)


def test_mean_attention_distance_all_first_token_is_position_dependent():
    """If every query position attends entirely to position 0, mean distance for
    query position i is exactly i -- averaged over all T positions, this has a known
    closed form: (0 + 1 + ... + (T-1)) / T = (T-1)/2."""
    T = 5
    weights = torch.zeros(1, T, T)
    weights[0, :, 0] = 1.0  # every query attends fully to key position 0
    distance = mean_attention_distance(weights)
    expected = (T - 1) / 2.0
    assert torch.allclose(distance, torch.tensor([expected]), atol=1e-5)


def test_summarize_attention_shapes():
    model = _tiny_model()
    T = 6
    input_ids = torch.randint(0, 100, (1, T))
    capture = capture_attention(model, input_ids, _FakeTokenizer(), device="cpu")
    summary = summarize_attention(capture)

    assert summary["n_layers"] == 2
    assert len(summary["entropy_by_layer"]) == 2
    assert len(summary["entropy_by_layer"][0]) == 4  # n_heads
    assert len(summary["distance_by_layer"]) == 2
    assert len(summary["distance_by_layer"][0]) == 4
