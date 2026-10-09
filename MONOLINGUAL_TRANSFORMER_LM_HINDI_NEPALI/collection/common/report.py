"""Renders the Phase 1 per-language Markdown report from computed statistics.

Pure string templating over dicts produced by :mod:`stats`, :mod:`tokenizer_utils`,
and the split manifest -- no computation happens here, so a report can be re-rendered
cheaply after any downstream number changes without recomputing anything.
"""

from __future__ import annotations

from pathlib import Path


def render_report(
    lang: str,
    corpus_stats: dict,
    split_manifest: dict,
    clean_funnel: dict,
    tokenizer_stats: dict,
    vocab_size: int,
    unk_rate: float,
    manual_fraction: float,
    independence: dict | None,
) -> str:
    """Build the full Markdown report body for one language."""
    lines: list[str] = [f"# Phase 1 report: {lang.capitalize()}", ""]

    lines += ["## 1. Corpus overview", ""]
    lines += [
        "| | Value |",
        "| --- | --- |",
        f"| Documents (cleaned) | {corpus_stats['docs']:,} |",
        f"| Characters | {corpus_stats['chars']:,} |",
        f"| Bytes | {corpus_stats['bytes']:,} |",
        "",
    ]

    lines += ["## 2. Manual vs. downloaded", ""]
    lines += ["| Method | Documents | Bytes | Share |", "| --- | --- | --- | --- |"]
    total_bytes = sum(corpus_stats["per_method_bytes"].values()) or 1
    for method, n_bytes in corpus_stats["per_method_bytes"].items():
        docs = corpus_stats["per_method_docs"][method]
        lines.append(f"| {method} | {docs:,} | {n_bytes:,} | {100*n_bytes/total_bytes:.2f}% |")
    lines += ["", f"**Manual-collection fraction (measured on final training tokens): {manual_fraction:.2%}**", ""]

    lines += ["## 3. Cleaning funnel", ""]
    lines += ["| Stage | Documents remaining |", "| --- | --- |"]
    for stage, count in clean_funnel.items():
        lines.append(f"| {stage} | {count:,} |")
    lines.append("")

    lines += ["## 4. Train/val/test split", ""]
    lines += ["| Split | Documents | Bytes |", "| --- | --- | --- |"]
    for name, info in split_manifest["splits"].items():
        lines.append(f"| {name} | {info['docs']:,} | {info['bytes']:,} |")
    lines.append("")

    lines += ["## 5. Tokenizer", ""]
    lines += [
        "SentencePiece Unigram, trained separately for this language (no shared vocabulary "
        "with the other language's model). Unigram was chosen over BPE because Hindi and "
        "Nepali are morphologically rich Indo-Aryan languages, and Unigram's likelihood-based "
        "pruning tends to align subword boundaries with morphemes better than BPE's greedy "
        "pairwise merges.",
        "",
        "| | Value |",
        "| --- | --- |",
        f"| Vocabulary size | {vocab_size:,} |",
        f"| Training tokens (exact, tokenizer-measured) | {tokenizer_stats['tokens']:,} |",
        f"| Fertility (chars/token) | {tokenizer_stats['fertility_chars_per_token']:.4f} |",
        f"| UNK rate (no-byte-fallback diagnostic) | {unk_rate:.2%} |",
        "",
    ]

    if independence is not None:
        lines += ["## 6. Corpus independence", ""]
        lines += [
            f"Shared documents with the other language's cleaned corpus: "
            f"**{independence['shared_docs']}** "
            f"({'independent' if independence['independent'] else 'NOT independent -- investigate'}).",
            "",
        ]

    return "\n".join(lines)


def write_report(text: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
