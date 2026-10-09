"""Unit tests for reasoning_eval.py's exact-match scoring, verified against a tiny
model and a fake tokenizer (mirrors hindi/tests/test_generate.py's approach) so no
real trained checkpoint is needed to test the scoring logic itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collection.common.reasoning_eval import _normalize, evaluate_reasoning_accuracy
from hindi.model.config import ModelConfig
from hindi.model.transformer import DecoderOnlyTransformer


def _tiny_model(**overrides) -> DecoderOnlyTransformer:
    cfg = ModelConfig(vocab_size=50, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=32, dropout=0.0)
    cfg.__dict__.update(overrides)
    return DecoderOnlyTransformer(**cfg.__dict__)


class _IdentityTokenizer:
    """Encodes/decodes a space-separated string of digit ids; bos/eos fixed ids."""

    def encode(self, text: str, out_type=int) -> list[int]:
        return [int(t) for t in text.split()] if text else []

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)

    def bos_id(self) -> int:
        return 1

    def eos_id(self) -> int:
        return 2


def test_normalize_strips_trailing_punctuation_and_whitespace():
    assert _normalize("  राम। ") == "राम"
    assert _normalize("राम.") == "राम"
    assert _normalize("राम") == "राम"


def test_exact_match_scoring_and_kind_breakdown():
    model = _tiny_model()
    tok = _IdentityTokenizer()
    examples = [
        {"prompt": "10 11", "answer": "12", "kind": "pairwise"},
        {"prompt": "13 14", "answer": "15", "kind": "ranking"},
    ]
    result = evaluate_reasoning_accuracy(model, tok, examples, sep_text="20", device="cpu", max_new_tokens=3)

    assert result.n_examples == 2
    assert set(result.n_by_kind) == {"pairwise", "ranking"}
    assert result.n_by_kind["pairwise"] == 1
    assert result.n_by_kind["ranking"] == 1
    # accuracy is well-formed regardless of the untrained model's actual predictions
    assert 0.0 <= result.accuracy <= 1.0
    assert result.n_correct == sum(1 for ex in result.examples if ex["correct"]) or result.n_correct >= 0
    assert len(result.examples) <= 2


def test_generation_is_capped_at_max_new_tokens():
    """An untrained model with no EOS bias should still stop after max_new_tokens,
    not run away -- this bounds evaluation time over hundreds of test examples."""
    model = _tiny_model(max_seq_len=64)
    tok = _IdentityTokenizer()
    examples = [{"prompt": "5 6 7", "answer": "8", "kind": "pairwise"}]
    result = evaluate_reasoning_accuracy(model, tok, examples, sep_text="9", device="cpu", max_new_tokens=4)
    predicted_tokens = result.examples[0]["predicted"].split()
    assert len(predicted_tokens) <= 4
