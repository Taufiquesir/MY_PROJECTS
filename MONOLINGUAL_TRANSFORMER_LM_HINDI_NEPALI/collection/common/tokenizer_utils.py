"""SentencePiece Unigram tokenizer training, vocab sweep, and exact token counting.

Unigram (not BPE) is used for both languages: Hindi and Nepali are morphologically
rich Indo-Aryan languages written in Devanagari, and Unigram's likelihood-based
subword pruning tends to recover morpheme-like units (case markers, verb endings)
better than BPE's greedy pairwise merges, which is the property that matters most for
a from-scratch LM with a small (tens-of-thousands) vocabulary budget.

``byte_fallback=True`` is used so a true out-of-vocabulary token is structurally
impossible -- any byte sequence can be represented as a fallback byte token even if it
never appears in the training corpus. A genuine UNK-rate diagnostic is still reported
by training a *separate* no-fallback tokenizer at the same vocab size purely to
measure how often it would have needed one; that diagnostic tokenizer is never shipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import sentencepiece as spm

from .record import iter_records


def export_plain_text(clean_dir: Path, out_path: Path, max_docs: int | None = None) -> int:
    """Write one document per line to a plain-text file for SentencePiece to train on.

    Returns the number of documents written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for rec in iter_records(clean_dir):
            line = rec.text.replace("\n", " ").strip()
            if not line:
                continue
            f.write(line + "\n")
            n += 1
            if max_docs is not None and n >= max_docs:
                break
    return n


def train_tokenizer(
    input_path: Path,
    model_prefix: Path,
    vocab_size: int,
    byte_fallback: bool = True,
    input_sentence_size: int = 3_000_000,
) -> None:
    """Train a SentencePiece Unigram tokenizer.

    ``normalization_rule_name="identity"`` is used because the corpus is already
    NFC-normalized by the cleaning pipeline; letting SentencePiece also apply its
    default NFKC would silently fold distinctions the cleaning step deliberately kept.
    """
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        byte_fallback=byte_fallback,
        normalization_rule_name="identity",
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        character_coverage=0.9995,
        num_threads=max(1, __import__("os").cpu_count() or 1),
    )


def count_tokens(model_path: Path, split_dir: Path) -> dict:
    """Run a trained tokenizer over an entire split and return exact stats.

    This is the *only* source of truth for reported token counts in this project --
    never an estimated bytes-per-token planning figure.
    """
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    total_tokens = 0
    total_chars = 0
    total_docs = 0
    for rec in iter_records(split_dir):
        ids = sp.encode(rec.text, out_type=int)
        total_tokens += len(ids)
        total_chars += len(rec.text)
        total_docs += 1
    return {
        "docs": total_docs,
        "chars": total_chars,
        "tokens": total_tokens,
        "fertility_chars_per_token": total_chars / total_tokens if total_tokens else 0.0,
    }


def measure_unk_rate(clean_dir: Path, vocab_size: int, scratch_dir: Path, sample_docs: int = 50_000) -> float:
    """Train a throwaway no-byte-fallback tokenizer to measure a genuine UNK rate.

    Returns the fraction of tokens that would be UNK without byte-fallback rescue.
    Never used as the shipped tokenizer.
    """
    scratch_dir.mkdir(parents=True, exist_ok=True)
    sample_path = scratch_dir / "unk_probe.txt"
    export_plain_text(clean_dir, sample_path, max_docs=sample_docs)
    prefix = scratch_dir / "unk_probe"
    train_tokenizer(sample_path, prefix, vocab_size, byte_fallback=False, input_sentence_size=sample_docs)
    sp = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
    unk_id = sp.unk_id()
    total, unk = 0, 0
    for rec in iter_records(clean_dir):
        ids = sp.encode(rec.text, out_type=int)
        total += len(ids)
        unk += sum(1 for i in ids if i == unk_id)
    return unk / total if total else 0.0


def sweep_vocab_sizes(train_sample_path: Path, val_dir: Path, sizes: list[int], scratch_dir: Path) -> dict:
    """Train a tokenizer at each candidate vocab size and score it on held-out val text.

    Returns ``{size: {"fertility": ..., "chars_per_token": ...}}`` so a vocab size can
    be picked from the fertility/coverage curve rather than guessed.
    """
    results = {}
    for size in sizes:
        prefix = scratch_dir / f"sweep_{size}"
        train_tokenizer(train_sample_path, prefix, size)
        stats = count_tokens(Path(str(prefix) + ".model"), val_dir)
        results[size] = stats
        print(f"[sweep] vocab={size}: fertility={stats['fertility_chars_per_token']:.4f} chars/token", flush=True)
    return results


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
