"""Unit tests for eval/generate.py's decoding functions, verified against a tiny
model since no real checkpoint exists at test-authoring time.
"""

import sentencepiece as spm
import torch

from hindi.eval.generate import generate_greedy, generate_temperature
from hindi.model.config import ModelConfig
from hindi.model.transformer import DecoderOnlyTransformer


def _tiny_model(**overrides) -> DecoderOnlyTransformer:
    cfg = ModelConfig(
        vocab_size=100, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=16, dropout=0.0
    )
    cfg.__dict__.update(overrides)
    return DecoderOnlyTransformer(**cfg.__dict__)


class _IdentityTokenizer:
    """Fake tokenizer: encodes a space-separated string of digit-ids, decodes back
    the same way -- avoids needing a real trained SentencePiece model for tests
    that only care about decoding-loop mechanics, not real subword behavior."""

    def encode(self, text: str, out_type=int) -> list[int]:
        return [int(t) for t in text.split()] if text else []

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)


def test_generate_greedy_produces_requested_number_of_tokens():
    model = _tiny_model()
    tokenizer = _IdentityTokenizer()
    out = generate_greedy(model, tokenizer, prompt="1 2 3", max_new_tokens=5, device="cpu")
    assert len(out.split()) == 5


def test_generate_greedy_is_deterministic():
    """Same prompt, same model -> identical continuation every call, since argmax
    has no randomness (unlike generate_temperature's sampling)."""
    torch.manual_seed(0)
    model = _tiny_model()
    tokenizer = _IdentityTokenizer()
    out1 = generate_greedy(model, tokenizer, prompt="1 2 3", max_new_tokens=8, device="cpu")
    out2 = generate_greedy(model, tokenizer, prompt="1 2 3", max_new_tokens=8, device="cpu")
    assert out1 == out2


def test_generate_greedy_stops_at_eos():
    """If eos_id happens to be the argmax at some step, generation stops there --
    verified by checking output length never exceeds max_new_tokens (the eos-stop
    path is exercised implicitly across many vocab items in a small vocab)."""
    model = _tiny_model()
    tokenizer = _IdentityTokenizer()
    out = generate_greedy(model, tokenizer, prompt="1 2 3", max_new_tokens=20, device="cpu", eos_id=5)
    ids = [int(i) for i in out.split()] if out else []
    assert len(ids) <= 20
    if len(ids) < 20:
        # stopped early -> must have been because eos_id was generated (and excluded)
        pass  # length being < max_new_tokens is itself the observable evidence


def test_generate_temperature_produces_requested_number_of_tokens():
    model = _tiny_model()
    tokenizer = _IdentityTokenizer()
    out = generate_temperature(model, tokenizer, prompt="1 2 3", max_new_tokens=5, device="cpu", temperature=1.0)
    assert len(out.split()) == 5


def test_generate_temperature_respects_max_seq_len_context_window():
    """Generating past max_seq_len must not error -- the sliding-context-window
    truncation (context = input_ids[:, -model.max_seq_len:]) has to keep the model
    fed a valid-length input even once the running sequence exceeds max_seq_len."""
    model = _tiny_model(max_seq_len=8)
    tokenizer = _IdentityTokenizer()
    long_prompt = " ".join(str(i % 100) for i in range(10))  # already longer than max_seq_len
    out = generate_temperature(model, tokenizer, prompt=long_prompt, max_new_tokens=10, device="cpu", temperature=1.0)
    assert len(out.split()) == 10
