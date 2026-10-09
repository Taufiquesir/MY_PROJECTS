"""Unit tests for the token-window dataset (not the tokenize-from-corpus step, which
needs a real trained tokenizer and real corpus data -- exercised manually against the
real Hindi/Nepali corpora, not in the fast unit-test suite).
"""

import numpy as np
import pytest

from hindi.model.data import TokenWindowDataset


@pytest.fixture
def tiny_token_stream(tmp_path):
    """A small deterministic uint16 token stream, written the same way
    tokenize_split_to_binary would."""
    path = tmp_path / "tokens.bin"
    arr = np.arange(1000, dtype=np.uint16)  # tokens 0, 1, 2, ..., 999
    arr.tofile(path)
    return path


def test_dataset_length_accounts_for_seq_len(tiny_token_stream):
    ds = TokenWindowDataset(tiny_token_stream, seq_len=10)
    assert len(ds) == 1000 - 10


def test_target_is_input_shifted_by_one(tiny_token_stream):
    ds = TokenWindowDataset(tiny_token_stream, seq_len=10)
    x, y = ds[5]
    # tokens were written as 0..999 in order, so window starting at offset 5 is
    # input=[5..14], target=[6..15] -- deterministic, exact values checkable.
    assert x.tolist() == list(range(5, 15))
    assert y.tolist() == list(range(6, 16))


def test_too_short_stream_raises(tmp_path):
    path = tmp_path / "short.bin"
    np.arange(5, dtype=np.uint16).tofile(path)
    with pytest.raises(ValueError):
        TokenWindowDataset(path, seq_len=10)
