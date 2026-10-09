"""End-to-end Phase 1 build: clean -> split -> tokenizer sweep -> train -> report.

Assumes raw collection (``collection.collect --stage news/wiki/public``) has already
been run for each language. This script picks up from there::

    python run_phase1.py --langs hindi nepali
    python run_phase1.py --langs hindi --skip sweep   # re-run everything but the sweep

On this machine (~7GB RAM, no swap headroom to spare), SentencePiece's Unigram trainer
is pointed at a pre-shrunk sample of the training split rather than the full corpus --
see ``TRAIN_SAMPLE_DOCS`` below -- because the trainer preprocesses its entire input
file before ``input_sentence_size`` subsampling takes effect, and doing that against a
multi-GB export reliably gets OOM-killed here regardless of thread count.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from collection.collect import data_root
from collection.common.clean import clean_corpus
from collection.common.plots import (
    plot_fertility_sweep,
    plot_filter_funnel,
    plot_length_histogram,
    plot_manual_vs_downloaded,
    plot_source_contribution,
)
from collection.common.report import render_report, write_report
from collection.common.split import split_corpus
from collection.common.stats import compute_corpus_stats, corpus_independence, write_json
from collection.common.tokenizer_utils import (
    count_tokens,
    export_plain_text,
    measure_unk_rate,
    sweep_vocab_sizes,
    train_tokenizer,
)

VOCAB_SIZES_TO_SWEEP = [5_000, 7_000, 9_000, 16_000, 24_000, 32_000]
SHIPPED_VOCAB_SIZE = 16_000  # placeholder; run_language() overrides this per-language from the sweep result
#: Lowered from 1,500,000 -> 500,000 -> 250,000 after repeated OOM kills at ~5.4GB
#: RSS regardless of sample size in that range (confirmed via `journalctl -k`) --
#: see collection/collect.py's own TRAIN_SAMPLE_DOCS for the full history. Keep in
#: sync with that copy.
TRAIN_SAMPLE_DOCS = 250_000

STAGES = ["clean", "split", "sweep", "train", "count", "report"]


def run_language(lang: str, skip: set[str]) -> None:
    root = data_root() / lang
    report_dir = Path("report/phase1")
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if "clean" not in skip:
        input_dirs = [d for d in (root / "manual_news", root / "manual_wiki", root / "public") if d.exists()]
        clean_stats = clean_corpus(input_dirs, root / "clean", lang=lang)
        write_json(
            {
                "seen": clean_stats.seen,
                "kept": clean_stats.kept,
                "dropped_short": clean_stats.dropped_short,
                "dropped_script": clean_stats.dropped_script,
                "dropped_repetitive": clean_stats.dropped_repetitive,
                "dropped_profanity": clean_stats.dropped_profanity,
                "dropped_contaminated": clean_stats.dropped_contaminated,
                "dropped_duplicate": clean_stats.dropped_duplicate,
                "dropped_near_duplicate": clean_stats.dropped_near_duplicate,
            },
            root / "clean_stats.json",
        )
        print(f"[{lang}] clean: kept {clean_stats.kept:,} of {clean_stats.seen:,}", flush=True)

    if "split" not in skip:
        split_corpus(root / "clean", root / "splits", root / "splits.json")

    sample_path = root / "tokenizer_sample.txt"
    if "sweep" not in skip or "train" not in skip:
        export_plain_text(root / "splits" / "train", sample_path, max_docs=TRAIN_SAMPLE_DOCS)

    sweep_dir = root / "tokenizer_sweep"
    if "sweep" not in skip:
        sweep_dir.mkdir(parents=True, exist_ok=True)
        results = sweep_vocab_sizes(sample_path, root / "splits" / "val", VOCAB_SIZES_TO_SWEEP, sweep_dir)
        write_json({str(k): v for k, v in results.items()}, sweep_dir / "sweep_results.json")
        plot_fertility_sweep(results, plots_dir / f"fertility_sweep_{lang}.png", lang)

    tok_dir = Path(lang) / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = tok_dir / f"{lang}_tokenizer"
    if "train" not in skip:
        train_tokenizer(sample_path, model_prefix, SHIPPED_VOCAB_SIZE)

    if "count" not in skip:
        model_path = Path(str(model_prefix) + ".model")
        tok_stats = count_tokens(model_path, root / "splits" / "train")
        write_json(tok_stats, root / "token_count.json")
        unk_rate = measure_unk_rate(root / "clean", SHIPPED_VOCAB_SIZE, root / "unk_probe")
        write_json({"unk_rate": unk_rate}, root / "unk_rate.json")
        print(f"[{lang}] tokens: {tok_stats['tokens']:,} (fertility {tok_stats['fertility_chars_per_token']:.4f}), UNK rate {unk_rate:.2%}", flush=True)

    if "report" not in skip:
        import json

        corpus_stats = compute_corpus_stats(root / "clean")
        split_manifest = json.loads((root / "splits.json").read_text())
        clean_funnel_raw = json.loads((root / "clean_stats.json").read_text())
        remaining = clean_funnel_raw["seen"]
        clean_funnel = {"seen": remaining}
        for stage_label, drop_key in (
            ("after length filter", "dropped_short"),
            ("after script filter", "dropped_script"),
            ("after repetition filter", "dropped_repetitive"),
            ("after profanity filter", "dropped_profanity"),
            ("after decontamination", "dropped_contaminated"),
            ("after exact dedup", "dropped_duplicate"),
            ("after near-dedup (kept)", "dropped_near_duplicate"),
        ):
            remaining -= clean_funnel_raw.get(drop_key, 0)
            clean_funnel[stage_label] = remaining
        tok_stats = json.loads((root / "token_count.json").read_text())
        unk_rate = json.loads((root / "unk_rate.json").read_text())["unk_rate"]
        manual_bytes = corpus_stats.per_method_bytes.get("manual", 0)
        downloaded_bytes = corpus_stats.per_method_bytes.get("public", 0)
        manual_fraction = manual_bytes / (manual_bytes + downloaded_bytes) if (manual_bytes + downloaded_bytes) else 0.0

        plot_source_contribution(corpus_stats.per_domain_bytes, plots_dir / f"source_contribution_{lang}.png", lang)
        plot_manual_vs_downloaded(manual_bytes, downloaded_bytes, plots_dir / f"manual_vs_downloaded_{lang}.png", lang)
        plot_filter_funnel(clean_funnel, plots_dir / f"filter_funnel_{lang}.png", lang)

        text = render_report(
            lang=lang,
            corpus_stats=corpus_stats.as_dict(),
            split_manifest=split_manifest,
            clean_funnel=clean_funnel,
            tokenizer_stats=tok_stats,
            vocab_size=SHIPPED_VOCAB_SIZE,
            unk_rate=unk_rate,
            manual_fraction=manual_fraction,
            independence=None,
        )
        write_report(text, report_dir / f"stats_{lang}.md")
        print(f"[{lang}] report written to report/phase1/stats_{lang}.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--langs", nargs="+", default=["hindi", "nepali"])
    parser.add_argument("--skip", nargs="*", default=[], choices=STAGES)
    args = parser.parse_args()
    skip = set(args.skip)
    for lang in args.langs:
        run_language(lang, skip)

    if len(args.langs) == 2 and "report" not in skip:
        a, b = args.langs
        independence = corpus_independence(data_root() / a / "clean", data_root() / b / "clean")
        write_json(independence, Path("report/phase1/corpus_independence.json"))
        print(f"Corpus independence ({a} vs {b}): {independence}", flush=True)


if __name__ == "__main__":
    main()
