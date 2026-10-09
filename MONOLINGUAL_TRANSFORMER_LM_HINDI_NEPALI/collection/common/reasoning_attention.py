"""Post-finetune attention analysis: pretrained vs. finetuned heatmaps/entropy/
distance on comparative-reasoning prompts, for one language.

Reuses Phase 2's attention toolkit (``<lang>/eval/attention_analysis.py``,
``<lang>/eval/plots.py``) completely unmodified -- ``capture_attention`` only needs
a model and a token sequence, so pointing it at a Phase 3 finetuned checkpoint
instead of the Phase 2 pretrained one requires no new capture/analysis code, only a
driver that loads both checkpoints and calls the same functions twice. This mirrors
the assignment's own framing ("reuse the Phase 2 attention toolkit on the finetuned
checkpoints").

Deliberately evaluated on comparative-reasoning prompts (drawn straight from
``<lang>/reasoning/splits/test.jsonl``) rather than the Phase 2 news-text examples
-- the assignment specifically asks to comment on attention "especially on
comparative-reasoning prompts", since that is the behavior finetuning was meant to
change, if it changed anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .reasoning_data import load_examples


def run_comparison(
    pretrained_checkpoint: Path,
    finetuned_checkpoint: Path,
    tokenizer_path: Path,
    test_split_path: Path,
    sep_text: str,
    lang: str,
    out_dir: Path,
    load_model_and_tokenizer_fn,
    capture_attention_fn,
    summarize_attention_fn,
    plot_attention_heatmap_multihead_fn,
    plot_entropy_by_layer_fn,
    plot_distance_by_layer_fn,
    n_examples: int = 2,
) -> dict:
    """Run attention capture on ``n_examples`` reasoning prompts through both the
    pretrained and finetuned checkpoints, writing heatmaps (early + late layer, all
    heads) and entropy/distance-by-layer plots for each, labeled so the two are
    visually distinguishable in the report. Returns a JSON-serializable summary
    dict with both checkpoints' entropy/distance numbers for direct comparison.
    """
    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    test_examples = load_examples(test_split_path)[:n_examples]
    # Full prompt fed to the model exactly as finetuning saw it (prompt + sep_text)
    # so the finetuned checkpoint's attention is inspected on the same input shape
    # it was actually trained on, not a truncated or differently-formatted prompt.
    prompts = [f"{ex['prompt']} {sep_text}" for ex in test_examples]

    summary: dict = {"lang": lang, "examples": [ex["prompt"] for ex in test_examples], "checkpoints": {}}

    for stage, ckpt_path in (("pretrained", pretrained_checkpoint), ("finetuned", finetuned_checkpoint)):
        model, tokenizer, device = load_model_and_tokenizer_fn(ckpt_path, tokenizer_path)
        stage_summaries = []

        for i, prompt in enumerate(prompts):
            ids = tokenizer.encode(prompt, out_type=int)[: model.max_seq_len]
            input_ids = torch.tensor([ids], dtype=torch.long)
            capture = capture_attention_fn(model, input_ids, tokenizer, device)
            attn_summary = summarize_attention_fn(capture)
            stage_summaries.append(attn_summary)

            n_heads = capture.weights[0].shape[0]
            all_heads = list(range(n_heads))
            plot_attention_heatmap_multihead_fn(
                capture, layer_idx=0, head_indices=all_heads, lang=f"{lang}-{stage}",
                out_path=plots_dir / f"attn_heatmap_{stage}_example{i}_layer0_allheads.png",
            )
            plot_attention_heatmap_multihead_fn(
                capture, layer_idx=len(capture.weights) - 1, head_indices=all_heads, lang=f"{lang}-{stage}",
                out_path=plots_dir / f"attn_heatmap_{stage}_example{i}_layerlast_allheads.png",
            )

        avg_entropy_by_layer = np.mean([s["entropy_by_layer"] for s in stage_summaries], axis=0).tolist()
        avg_distance_by_layer = np.mean([s["distance_by_layer"] for s in stage_summaries], axis=0).tolist()
        plot_entropy_by_layer_fn(avg_entropy_by_layer, f"{lang}-{stage}", plots_dir / f"attention_entropy_by_layer_{stage}.png")
        plot_distance_by_layer_fn(avg_distance_by_layer, f"{lang}-{stage}", plots_dir / f"attention_distance_by_layer_{stage}.png")

        summary["checkpoints"][stage] = {
            "checkpoint": str(ckpt_path),
            "avg_entropy_by_layer": avg_entropy_by_layer,
            "avg_distance_by_layer": avg_distance_by_layer,
        }
        del model  # free before loading the next checkpoint, same RAM concern as reasoning_eval.py

    out_path = out_dir / "attention_comparison.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
