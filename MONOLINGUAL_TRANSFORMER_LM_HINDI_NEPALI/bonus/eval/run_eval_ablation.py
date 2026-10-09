"""End-to-end evaluation for the bonus no-positional-embedding ablation: the
exact same Phase 2 suite (PPL/BPB -> generation quality/diversity at greedy +
temperature 0.5/1.0/1.5 -> attention analysis) run against the ablated
checkpoint, per the bonus spec's "run the full Phase 2 evaluation suite on that
ablated model" requirement.

Structurally a copy of the top-level ``run_eval.py``, with every module it
imports swapped for this folder's ablation counterpart (``generate_ablation``,
``perplexity_ablation``, ``quality_ablation``, ``attention_analysis_ablation``,
``plots_ablation``) instead of ``hindi.eval.*``. Writes everything under
``bonus/report/``: ``metrics.json`` (all numbers) and every plot as a ``.png``
under ``bonus/report/plots/`` -- so the ablation's own results live entirely
inside ``bonus/``, never mixed into ``report/phase2/hindi/``.

Usage (from the repo root)::

    python -m bonus.eval.run_eval_ablation --checkpoint bonus/checkpoints/best.pt --test-bin /path/to/test.bin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from collection.common.record import iter_records  # noqa: E402

from bonus.eval.attention_analysis_ablation import capture_attention, summarize_attention  # noqa: E402
from bonus.eval.generate_ablation import load_model_and_tokenizer  # noqa: E402
from bonus.eval.perplexity_ablation import evaluate_perplexity, measure_bytes_per_token  # noqa: E402
from bonus.eval.plots_ablation import (  # noqa: E402
    plot_attention_heatmap_multihead,
    plot_distance_by_layer,
    plot_entropy_by_layer,
)
from bonus.eval.quality_ablation import evaluate_quality  # noqa: E402

LANG = "hindi"  # the ablation is on Model H (Hindi) -- see bonus/README.md


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "bonus" / "checkpoints" / "best.pt")
    parser.add_argument("--splits-dir", type=Path, default=REPO_ROOT / "data" / LANG / "splits")
    parser.add_argument("--tokenizer-path", type=Path, default=REPO_ROOT / LANG / "tokenizer" / f"{LANG}_tokenizer.model")
    parser.add_argument("--test-bin", type=Path, required=True, help="tokenized test split .bin, from hindi.model.data.tokenize_split_to_binary")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "bonus" / "report")
    parser.add_argument("--n-quality-samples", type=int, default=50)
    parser.add_argument("--n-heatmap-examples", type=int, default=2)
    parser.add_argument("--ppl-max-batches", type=int, default=2000)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ablation] loading model + tokenizer from {args.checkpoint}", flush=True)
    model, tokenizer, device = load_model_and_tokenizer(args.checkpoint, args.tokenizer_path)
    print(f"[ablation] device: {device}, params: {model.num_parameters():,}", flush=True)

    # --- PPL / BPB ---
    print("[ablation] measuring bytes/token for BPB ...", flush=True)
    bytes_per_token = measure_bytes_per_token(args.test_bin, args.splits_dir / "test", args.tokenizer_path)
    print("[ablation] evaluating perplexity/BPB over test split ...", flush=True)
    ppl_result = evaluate_perplexity(
        args.checkpoint, args.test_bin, bytes_per_token, device=device, max_batches=args.ppl_max_batches
    )
    print(
        f"[ablation] PPL={ppl_result.ppl:.2f} BPB={ppl_result.bpb:.4f} "
        f"(over {ppl_result.n_tokens_evaluated:,} tokens)",
        flush=True,
    )

    # --- Generation quality + diversity: greedy + temperature 0.5/1.0/1.5 ---
    print("[ablation] loading test-split documents for generation quality eval ...", flush=True)
    test_records = [rec.text for rec in iter_records(args.splits_dir / "test")]
    print(f"[ablation] generating and scoring {args.n_quality_samples} samples x 4 decoding conditions (slow step) ...", flush=True)
    quality_result = evaluate_quality(
        args.checkpoint, args.tokenizer_path, test_records, n_samples=args.n_quality_samples, device=device
    )
    for name, cond in quality_result.by_condition.items():
        print(
            f"[ablation] [{name}] BLEU={cond.bleu:.2f} chrF={cond.chrf:.2f} "
            f"ROUGE-L={cond.rouge_l_f1:.4f} Distinct-1={cond.distinct_1:.4f} "
            f"Distinct-2={cond.distinct_2:.4f} repetition_rate={cond.repetition_rate:.4f}",
            flush=True,
        )

    # --- Attention analysis ---
    print(f"[ablation] running attention analysis on {args.n_heatmap_examples} example sequences ...", flush=True)
    attention_summaries = []
    for i, text in enumerate(test_records[: args.n_heatmap_examples]):
        ids = tokenizer.encode(text, out_type=int)[: model.max_seq_len]
        input_ids = torch.tensor([ids], dtype=torch.long)
        capture = capture_attention(model, input_ids, tokenizer, device)
        summary = summarize_attention(capture)
        attention_summaries.append(summary)

        n_heads = capture.weights[0].shape[0]
        all_heads = list(range(n_heads))
        plot_attention_heatmap_multihead(
            capture, layer_idx=0, head_indices=all_heads, lang="hindi-ablation",
            out_path=out_dir / "plots" / f"attn_heatmap_example{i}_layer0_allheads.png",
        )
        plot_attention_heatmap_multihead(
            capture, layer_idx=len(capture.weights) - 1, head_indices=all_heads, lang="hindi-ablation",
            out_path=out_dir / "plots" / f"attn_heatmap_example{i}_layerlast_allheads.png",
        )

    avg_entropy_by_layer = np.mean([s["entropy_by_layer"] for s in attention_summaries], axis=0).tolist()
    avg_distance_by_layer = np.mean([s["distance_by_layer"] for s in attention_summaries], axis=0).tolist()
    plot_entropy_by_layer(avg_entropy_by_layer, "hindi-ablation", out_dir / "plots" / "attention_entropy_by_layer.png")
    plot_distance_by_layer(avg_distance_by_layer, "hindi-ablation", out_dir / "plots" / "attention_distance_by_layer.png")

    # --- Write everything to metrics.json for the report ---
    metrics = {
        "lang": "hindi-ablation",
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
                    for s in cond.samples[:5]
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
    print(f"[ablation] wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
