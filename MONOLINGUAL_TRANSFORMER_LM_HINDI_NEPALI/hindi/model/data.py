"""Turns a Phase 1 split (JSONL shards of cleaned documents) into fixed-length token
windows for causal language model training.

Two-stage pipeline, run once per split before training:
  1. :func:`tokenize_split_to_binary` reads every document in a split, encodes it
     with the language's own trained SentencePiece tokenizer, and concatenates all
     token ids (with an end-of-document separator) into one flat binary file of
     ``uint16`` ids on disk -- vocab size is 9,000 for both languages, well under
     uint16's 65,535 range, so 2 bytes/token instead of the 8 bytes a Python int or
     int64 tensor element would cost. This avoids re-tokenizing text on every
     training epoch and avoids holding the whole tokenized corpus in RAM at once.
  2. :class:`TokenWindowDataset` memory-maps that binary file and serves random
     fixed-length windows from it -- ``np.memmap`` means the OS pages the file in
     from disk on demand rather than the dataset process holding hundreds of MB to
     GB of token ids resident in RAM, which matters on both this project's
     memory-constrained local machine and a shared Kaggle GPU instance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import Dataset


def tokenize_split_to_binary(
    split_dir: Path,
    tokenizer_path: Path,
    out_path: Path,
    eos_id: int | None = None,
    flush_every: int = 200_000,
) -> int:
    """Tokenize every document in ``split_dir`` and write a flat ``uint16`` token
    stream to ``out_path``. Returns the total number of tokens written.

    Documents are separated by ``eos_id`` (the tokenizer's own end-of-sentence id,
    if it has one) so a training window never silently concatenates the end of one
    unrelated document with the start of another without *some* boundary signal --
    the model can then learn to treat that id as a natural stopping point, similar
    to how GPT-style corpora are packed in practice.

    Writes incrementally in ``flush_every``-token chunks rather than accumulating
    every token id from the whole split into one Python list before writing once --
    for a large corpus (Nepali's ~568M tokens) that list's per-int object overhead
    was large enough to OOM-kill a memory-constrained session (observed: silent
    kernel restart on Colab's free tier partway through tokenizing a single split).
    Streaming keeps peak memory bounded to one chunk instead of the whole split.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from collection.common.record import iter_records

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if eos_id is None:
        eos_id = sp.eos_id()
        if eos_id < 0:  # tokenizer has no eos token defined
            eos_id = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    chunk: list[int] = []
    with open(out_path, "wb") as fh:
        for rec in iter_records(split_dir):
            ids = sp.encode(rec.text, out_type=int)
            chunk.extend(ids)
            if eos_id is not None:
                chunk.append(eos_id)
            if len(chunk) >= flush_every:
                arr = np.array(chunk, dtype=np.uint16)
                arr.tofile(fh)
                total += len(arr)
                chunk = []
        if chunk:
            arr = np.array(chunk, dtype=np.uint16)
            arr.tofile(fh)
            total += len(arr)
    return total


class TokenWindowDataset(Dataset):
    """Serves random fixed-length ``(input_ids, target_ids)`` windows from a
    memory-mapped flat token-id binary file, for causal LM training.

    ``target_ids`` is ``input_ids`` shifted by one position -- position ``t`` of
    ``input_ids`` pairs with the token that should follow it, which is exactly the
    "logits at position t predict the token at position t+1" objective the
    assignment specifies. A dataset of length ``N`` (in tokens) has
    ``N - seq_len`` possible starting offsets for a window of ``seq_len + 1`` tokens
    (the extra token gives the final target); this class treats each valid offset as
    one "example" for ``__len__``/``__getitem__`` purposes, though in practice
    training usually samples a bounded number of them per epoch rather than
    iterating every offset (see the training script).
    """

    def __init__(self, bin_path: Path, seq_len: int) -> None:
        self.seq_len = seq_len
        # dtype must match what tokenize_split_to_binary wrote (uint16).
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        if len(self.data) <= seq_len:
            raise ValueError(
                f"token stream ({len(self.data)} tokens) is shorter than seq_len+1 ({seq_len + 1})"
            )

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # +1 so there's a target for the last input position too.
        chunk = self.data[idx : idx + self.seq_len + 1]
        # np.memmap slices are read-only views; .astype(np.int64) both copies (so the
        # returned tensor owns its memory, safe to hand off across DataLoader worker
        # processes) and widens to a dtype PyTorch's embedding/cross-entropy ops
        # expect (they index with int64, not uint16).
        chunk = chunk.astype(np.int64)
        input_ids = torch.from_numpy(chunk[:-1])
        target_ids = torch.from_numpy(chunk[1:])
        return input_ids, target_ids
