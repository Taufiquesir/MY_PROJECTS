"""End-to-end Phase 2 evaluation: PPL/BPB -> generation quality/diversity (greedy +
temperature 0.5/1.0/1.5) -> attention analysis -> report tables/plots, for one
already-trained checkpoint.

Usage (from the repo root)::

    python run_eval.py --lang hindi --checkpoint hindi/train/checkpoints/best.pt --test-bin /path/to/test.bin
    python run_eval.py --lang nepali --checkpoint /path/to/best.pt --test-bin /path/to/test.bin --n-quality-samples 50

Assumes training has already produced a checkpoint (``model/checkpoint.py``'s
format) and that the language's ``splits/test`` directory and tokenizer are in
their usual repo-relative locations. Writes everything under
``report/phase2/<lang>/``: ``metrics.json`` (all numbers, for the report tables)
and every plot as a ``.png`` under ``plots/``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from collection.common.record import iter_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", required=True, choices=["hindi", "nepali"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=None, help="defaults to data/<lang>/splits")
    parser.add_argument("--tokenizer-path", type=Path, default=None, help="defaults to <lang>/tokenizer/<lang>_tokenizer.model")
    parser.add_argument("--test-bin", type=Path, required=True, help="tokenized test split .bin, from model.data.tokenize_split_to_binary")
    parser.add_argument("--out-dir", type=Path, default=None, help="defaults to report/phase2/<lang>")
    parser.add_argument("--n-quality-samples", type=int, default=50)
    parser.add_argument("--n-heatmap-examples", type=int, default=2, help="how many held-out sequences to render full attention heatmaps for")
    parser.add_argument(
        "--ppl-max-batches", type=int, default=2000,
        help="cap PPL/BPB evaluation to this many batches (None = every window in the test split, "
             "which for a multi-hundred-million-token corpus is hundreds of thousands of batches -- "
             "far more than needed for a stable estimate, and slow on CPU-only hardware)",
    )
    args = parser.parse_args()

    lang = args.lang
    # Each language's eval/ is a self-contained duplicate (per the assignment's
    # repository-layout example), not a shared top-level package -- so which
    # concrete module gets imported depends on --lang, resolved dynamically here
    # rather than a fixed top-of-file import statement.
    capture_attention = importlib.import_module(f"{lang}.eval.attention_analysis").capture_attention
    summarize_attention = importlib.import_module(f"{lang}.eval.attention_analysis").summarize_attention
    load_model_and_tokenizer = importlib.import_module(f"{lang}.eval.generate").load_model_and_tokenizer
    evaluate_perplexity = importlib.import_module(f"{lang}.eval.perplexity").evaluate_perplexity
    measure_bytes_per_token = importlib.import_module(f"{lang}.eval.perplexity").measure_bytes_per_token
    plots_mod = importlib.import_module(f"{lang}.eval.plots")
    plot_attention_heatmap_multihead = plots_mod.plot_attention_heatmap_multihead
    plot_distance_by_layer = plots_mod.plot_distance_by_layer
    plot_entropy_by_layer = plots_mod.plot_entropy_by_layer
    evaluate_quality = importlib.import_module(f"{lang}.eval.quality").evaluate_quality

    splits_dir = args.splits_dir or REPO_ROOT / "data" / lang / "splits"
    tokenizer_path = args.tokenizer_path or REPO_ROOT / lang / "tokenizer" / f"{lang}_tokenizer.model"
    out_dir = args.out_dir or REPO_ROOT / "report" / "phase2" / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{lang}] loading model + tokenizer from {args.checkpoint}", flush=True)
    model, tokenizer, device = load_model_and_tokenizer(args.checkpoint, tokenizer_path)
    print(f"[{lang}] device: {device}, params: {model.num_parameters():,}", flush=True)

    # --- PPL / BPB ---
    print(f"[{lang}] measuring bytes/token for BPB ...", flush=True)
    bytes_per_token = measure_bytes_per_token(args.test_bin, splits_dir / "test", tokenizer_path)
    print(f"[{lang}] evaluating perplexity/BPB over test split ...", flush=True)
    ppl_result = evaluate_perplexity(
        args.checkpoint, args.test_bin, bytes_per_token, device=device, max_batches=args.ppl_max_batches
    )
    print(
        f"[{lang}] PPL={ppl_result.ppl:.2f} BPB={ppl_result.bpb:.4f} "
        f"(over {ppl_result.n_tokens_evaluated:,} tokens)",
        flush=True,
    )

    # --- Generation quality + diversity: greedy + temperature 0.5/1.0/1.5 ---
    print(f"[{lang}] loading test-split documents for generation quality eval ...", flush=True)
    test_records = [rec.text for rec in iter_records(splits_dir / "test")]
    print(f"[{lang}] generating and scoring {args.n_quality_samples} samples x 4 decoding conditions (slow step) ...", flush=True)
    quality_result = evaluate_quality(
        args.checkpoint, tokenizer_path, test_records, n_samples=args.n_quality_samples, device=device
    )
    for name, cond in quality_result.by_condition.items():
        print(
            f"[{lang}] [{name}] BLEU={cond.bleu:.2f} chrF={cond.chrf:.2f} "
            f"ROUGE-L={cond.rouge_l_f1:.4f} Distinct-1={cond.distinct_1:.4f} "
            f"Distinct-2={cond.distinct_2:.4f} repetition_rate={cond.repetition_rate:.4f}",
            flush=True,
        )

    # --- Attention analysis ---
    print(f"[{lang}] running attention analysis on {args.n_heatmap_examples} example sequences ...", flush=True)
    attention_summaries = []
    for i, text in enumerate(test_records[: args.n_heatmap_examples]):
        ids = tokenizer.encode(text, out_type=int)[: model.max_seq_len]
        input_ids = torch.tensor([ids], dtype=torch.long)
        capture = capture_attention(model, input_ids, tokenizer, device)
        summary = summarize_attention(capture)
        attention_summaries.append(summary)

        # Multi-head heatmaps for one early layer (0) and one late layer (last),
        # all heads shown per layer -- per the assignment's explicit "at least one
        # early layer, one late layer, and multiple heads" requirement.
        n_heads = capture.weights[0].shape[0]
        all_heads = list(range(n_heads))
        plot_attention_heatmap_multihead(
            capture, layer_idx=0, head_indices=all_heads, lang=lang,
            out_path=out_dir / "plots" / f"attn_heatmap_example{i}_layer0_allheads.png",
        )
        plot_attention_heatmap_multihead(
            capture, layer_idx=len(capture.weights) - 1, head_indices=all_heads, lang=lang,
            out_path=out_dir / "plots" / f"attn_heatmap_example{i}_layerlast_allheads.png",
        )

    # Average entropy/distance across the example sequences for the summary plots.
    avg_entropy_by_layer = np.mean([s["entropy_by_layer"] for s in attention_summaries], axis=0).tolist()
    avg_distance_by_layer = np.mean([s["distance_by_layer"] for s in attention_summaries], axis=0).tolist()
    plot_entropy_by_layer(avg_entropy_by_layer, lang, out_dir / "plots" / "attention_entropy_by_layer.png")
    plot_distance_by_layer(avg_distance_by_layer, lang, out_dir / "plots" / "attention_distance_by_layer.png")

    # --- Write everything to metrics.json for the report ---
    metrics = {
        "lang": lang,
        "checkpoint": str(args.checkpoint),
        "n_params": model.num_parameters(),
        "perplexity": {
            "mean_nll": ppl_result.mean_nll,
            "ppl": ppl_result.ppl,
            "bpb": ppl_result.bpb,
            "n_tokens_evaluated": ppl_result.n_tokens_evaluated,
            "bytes_per_token": bytes_per_token,
        },
        "generation_quality": {
            name: {
                "bleu": cond.bleu,
                "chrf": cond.chrf,
                "rouge_l_f1": cond.rouge_l_f1,
                "distinct_1": cond.distinct_1,
                "distinct_2": cond.distinct_2,
                "repetition_rate": cond.repetition_rate,
                "examples": [
                    {"prompt": s.prompt, "reference": s.reference, "generated": s.generated}
                    for s in cond.samples[:5]  # a handful per condition for the report
                ],
            }
            for name, cond in quality_result.by_condition.items()
        },
        "n_quality_samples": quality_result.n_samples,
        "attention_avg_entropy_by_layer": avg_entropy_by_layer,
        "attention_avg_distance_by_layer": avg_distance_by_layer,
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{lang}] wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
