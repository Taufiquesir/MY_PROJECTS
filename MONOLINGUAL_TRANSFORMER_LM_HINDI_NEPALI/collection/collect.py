"""Phase 1 data-collection CLI.

Each stage can be run and re-run independently, so a long crawl can be interrupted
and resumed without repeating finished work::

    python -m collection.collect --lang hindi  --stage news
    python -m collection.collect --lang nepali --stage wiki
    python -m collection.collect --lang hindi  --stage public
    python -m collection.collect --lang hindi  --stage clean
    python -m collection.collect --lang hindi  --stage split
    python -m collection.collect --lang hindi  --stage sweep
    python -m collection.collect --lang hindi  --stage train
    python -m collection.collect --lang hindi  --stage count

``news`` and ``wiki`` both fill the manual-collection slice (>=20% of final training
tokens); ``public`` streams the downloaded slice from AI4Bharat Sangraha.

``sweep``/``train``/``count`` are each their own stage (not bundled into one combined
tokenizer command) so a long-running stage can be launched, waited on, and verified
against its own output file in isolation -- running clean/split/sweep/train together
under one process and trying to skip a subset with a flag is what caused a real bug:
``split``'s ``--skip`` list didn't include ``clean``, so ``clean`` silently re-ran and
(since neither stage cleared its own output directory before writing) doubled the
corpus. Every writer-based stage now clears its own output first, but staying
one-stage-per-command keeps failures small and easy to diagnose regardless.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from .common.crawler import SiteSpec, run_news_crawl

LANGUAGES = ("hindi", "nepali")
SANGRAHA_SUBSET = {"hindi": "verified/hin", "nepali": "verified/nep"}
WIKI_CODE = {"hindi": "hi", "nepali": "ne"}


def data_root() -> Path:
    """Corpus root, overridable via the ``DATA_ROOT`` env var (defaults to ``data``)."""
    return Path(os.environ.get("DATA_ROOT", "data"))


def _load_config(lang: str) -> tuple[list[SiteSpec], int, float]:
    path = Path(__file__).parent / f"sources_{lang}.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    specs = [SiteSpec(**entry) for entry in cfg["domains"]]
    return specs, int(cfg["target_tokens"]), float(cfg["bytes_per_token"])


def stage_news(lang: str, target_tokens: int | None) -> None:
    specs, default_target, bytes_per_token = _load_config(lang)
    tokens = target_tokens or default_target
    target_bytes = int(tokens * bytes_per_token)
    print(f"[{lang}] news crawl target ~{tokens:,} tokens (~{target_bytes/1e6:.0f} MB)", flush=True)
    budget = run_news_crawl(lang, specs, data_root(), target_bytes)
    print(f"[{lang}] news crawl stopped: {budget.docs_collected:,} docs, {budget.bytes_collected/1e6:.1f} MB", flush=True)


def stage_wiki(lang: str, target_tokens: int | None) -> None:
    from .common.wiki import run_wikipedia_ingest

    _, default_target, bytes_per_token = _load_config(lang)
    tokens = target_tokens or default_target
    target_bytes = int(tokens * bytes_per_token)
    collected = run_wikipedia_ingest(lang, WIKI_CODE[lang], data_root(), target_bytes)
    print(f"[{lang}] wikipedia stopped: {collected/1e6:.1f} MB collected this run", flush=True)


#: Manual (news + wiki) must end up at least this fraction of total raw bytes --
#: 5 points over the assignment's 20% floor, since cleaning tends to drop manual text
#: (near-dupes, boilerplate) at a different rate than public text, and the margin
#: needs to survive that shift, not just clear the floor on raw bytes.
MANUAL_FRACTION_TARGET = 0.25


def _measured_manual_bytes(lang: str) -> int:
    """Sum actual collected bytes across news + wiki, not the configured target.

    Used to size the public download against *real* progress: if a crawl is cut short
    before reaching its configured token target (e.g. by a deadline), sizing public
    off the configured target instead of what was actually collected would silently
    let manual's true share fall under the assignment's 20% floor.
    """
    from .common.record import iter_records

    root = data_root() / lang
    news_bytes = sum(r.n_bytes for r in iter_records(root / "manual_news"))
    wiki_bytes = sum(r.n_bytes for r in iter_records(root / "manual_wiki"))
    return news_bytes + wiki_bytes


def stage_public(lang: str, target_tokens: int | None) -> None:
    from .common.public_corpus import ingest_sangraha

    _, _, bytes_per_token = _load_config(lang)
    manual_bytes = _measured_manual_bytes(lang)
    if target_tokens is not None:
        target_bytes = int(target_tokens * bytes_per_token)
    else:
        # Solve for the public byte budget that keeps manual at exactly
        # MANUAL_FRACTION_TARGET of (manual + public), given manual bytes actually on
        # disk right now: public_bytes = manual_bytes * (1 - f) / f.
        target_bytes = int(manual_bytes * (1 - MANUAL_FRACTION_TARGET) / MANUAL_FRACTION_TARGET)
    print(
        f"[{lang}] public target sized from {manual_bytes/1e6:.1f} MB measured manual "
        f"bytes -> {target_bytes/1e6:.1f} MB public (manual fraction target "
        f"{MANUAL_FRACTION_TARGET:.0%})",
        flush=True,
    )
    ingest_sangraha(
        lang=lang,
        sangraha_dir=SANGRAHA_SUBSET[lang],
        out_dir=data_root() / lang / "public",
        target_bytes=target_bytes,
    )


def stage_clean(lang: str) -> None:
    from .common.clean import clean_corpus
    from .common.tokenizer_utils import write_json

    root = data_root() / lang
    input_dirs = [d for d in (root / "manual_news", root / "manual_wiki", root / "public") if d.exists()]
    stats = clean_corpus(input_dirs, root / "clean", lang=lang)
    print(f"[{lang}] cleaned {stats.kept:,} of {stats.seen:,} documents ({stats.output_bytes/1e6:.1f} MB)", flush=True)
    print(
        f"    dropped: short={stats.dropped_short:,} script={stats.dropped_script:,} "
        f"repetitive={stats.dropped_repetitive:,} profanity={stats.dropped_profanity:,} "
        f"contaminated={stats.dropped_contaminated:,} exact_dup={stats.dropped_duplicate:,} "
        f"near_dup={stats.dropped_near_duplicate:,}",
        flush=True,
    )
    for method, n in sorted(stats.per_method.items()):
        print(f"    {method}: {n:,} docs")
    write_json(
        {
            "seen": stats.seen,
            "kept": stats.kept,
            "dropped_short": stats.dropped_short,
            "dropped_script": stats.dropped_script,
            "dropped_repetitive": stats.dropped_repetitive,
            "dropped_profanity": stats.dropped_profanity,
            "dropped_contaminated": stats.dropped_contaminated,
            "dropped_duplicate": stats.dropped_duplicate,
            "dropped_near_duplicate": stats.dropped_near_duplicate,
        },
        root / "clean_stats.json",
    )


def stage_split(lang: str) -> None:
    from .common.split import split_corpus

    root = data_root() / lang
    manifest = split_corpus(root / "clean", root / "splits", root / "splits.json")
    for name, info in manifest["splits"].items():
        print(f"[{lang}/{name}] {info['docs']:,} docs, {info['bytes']/1e6:.1f} MB")


#: Candidate vocab sizes for the fertility/UNK sweep -- mirrors run_phase1.py's
#: VOCAB_SIZES_TO_SWEEP so the two entry points never drift apart.
VOCAB_SIZES_TO_SWEEP = [5_000, 7_000, 9_000, 16_000, 24_000, 32_000]
#: Cap on how many training documents SentencePiece ever sees directly. The Unigram
#: trainer preprocesses its *entire* input file before input_sentence_size
#: subsampling takes effect, so exporting the full multi-million-doc train split
#: straight to disk and pointing the trainer at it risks an OOM kill on this
#: machine regardless of thread count -- shrink the file itself first instead.
#: Lowered from 1,500,000 -> 500,000 after Hindi's sweep was OOM-killed twice in a
#: row at ~5.4GB RSS, then lowered again to 250,000 after Nepali's sweep was
#: *also* OOM-killed at ~5.4GB RSS even at 500,000 docs (confirmed via
#: `journalctl -k` both times, not a slow/stuck process). The RSS peak did not scale
#: down proportionally with document count between these two failures, which
#: suggests SentencePiece's fixed per-run overhead (vocabulary/trie construction,
#: not just per-sentence storage) dominates at this vocab-size range -- so cutting
#: the sample further is a safety margin, not a precisely-derived number.
TRAIN_SAMPLE_DOCS = 250_000


def stage_sweep(lang: str) -> None:
    from .common.tokenizer_utils import export_plain_text, sweep_vocab_sizes, write_json

    root = data_root() / lang
    sample_path = root / "tokenizer_sample.txt"
    n = export_plain_text(root / "splits" / "train", sample_path, max_docs=TRAIN_SAMPLE_DOCS)
    print(f"[{lang}] exported {n:,} docs to {sample_path} for tokenizer training", flush=True)
    sweep_dir = root / "tokenizer_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results = sweep_vocab_sizes(sample_path, root / "splits" / "val", VOCAB_SIZES_TO_SWEEP, sweep_dir)
    write_json({str(k): v for k, v in results.items()}, sweep_dir / "sweep_results.json")
    print(f"[{lang}] sweep results written to {sweep_dir / 'sweep_results.json'}", flush=True)


def stage_train(lang: str, target_tokens: int | None) -> None:
    from .common.tokenizer_utils import export_plain_text, train_tokenizer

    if target_tokens is None:
        raise SystemExit("--stage train requires --target-tokens <vocab_size> (the shipped vocab size, chosen from the sweep results)")
    vocab_size = target_tokens
    root = data_root() / lang
    sample_path = root / "tokenizer_sample.txt"
    if not sample_path.exists():
        export_plain_text(root / "splits" / "train", sample_path, max_docs=TRAIN_SAMPLE_DOCS)
    tok_dir = Path(lang) / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = tok_dir / f"{lang}_tokenizer"
    train_tokenizer(sample_path, model_prefix, vocab_size)
    print(f"[{lang}] trained tokenizer (vocab={vocab_size:,}) -> {model_prefix}.model", flush=True)


def stage_count(lang: str) -> None:
    import sentencepiece as spm

    from .common.tokenizer_utils import count_tokens, measure_unk_rate, write_json

    root = data_root() / lang
    model_path = Path(lang) / "tokenizer" / f"{lang}_tokenizer.model"
    if not model_path.exists():
        raise SystemExit(f"no trained tokenizer at {model_path} -- run --stage train first")
    tok_stats = count_tokens(model_path, root / "splits" / "train")
    write_json(tok_stats, root / "token_count.json")
    # measure_unk_rate trains its own throwaway probe tokenizer at this vocab size --
    # read the shipped model's actual vocab size so the probe matches it, rather than
    # guessing a default that could silently diverge from what was actually trained.
    shipped_vocab_size = spm.SentencePieceProcessor(model_file=str(model_path)).vocab_size()
    unk_rate = measure_unk_rate(root / "clean", shipped_vocab_size, root / "unk_probe")
    write_json({"unk_rate": unk_rate}, root / "unk_rate.json")
    print(
        f"[{lang}] tokens: {tok_stats['tokens']:,} (fertility {tok_stats['fertility_chars_per_token']:.4f} chars/token), "
        f"UNK rate {unk_rate:.2%}",
        flush=True,
    )


STAGES = {
    "news": stage_news,
    "wiki": stage_wiki,
    "public": stage_public,
    "clean": lambda lang, target: stage_clean(lang),
    "split": lambda lang, target: stage_split(lang),
    "sweep": lambda lang, target: stage_sweep(lang),
    "train": stage_train,
    "count": lambda lang, target: stage_count(lang),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", required=True, choices=LANGUAGES)
    parser.add_argument("--stage", required=True, choices=tuple(STAGES))
    parser.add_argument("--target-tokens", type=int, default=None)
    args = parser.parse_args()
    STAGES[args.stage](args.lang, args.target_tokens)


if __name__ == "__main__":
    main()
