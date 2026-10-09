"""Post-finetune attention comparison for Model H (Hindi): pretrained vs. finetuned
heatmaps/entropy/distance on comparative-reasoning prompts.

Usage (from the repo root)::

    python -m hindi.reasoning.attention_compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from collection.common.reasoning_attention import run_comparison  # noqa: E402
from hindi.eval.attention_analysis import capture_attention, summarize_attention  # noqa: E402
from hindi.eval.generate import load_model_and_tokenizer  # noqa: E402
from hindi.eval.plots import (  # noqa: E402
    plot_attention_heatmap_multihead,
    plot_distance_by_layer,
    plot_entropy_by_layer,
)

LANG = "hindi"
SEP_TEXT = "उत्तर:"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=REPO_ROOT / LANG / "train" / "checkpoints" / "best.pt")
    parser.add_argument("--finetuned-checkpoint", type=Path, default=REPO_ROOT / LANG / "reasoning" / "checkpoints" / "best.pt")
    parser.add_argument("--tokenizer-path", type=Path, default=REPO_ROOT / LANG / "tokenizer" / f"{LANG}_tokenizer.model")
    parser.add_argument("--test-split", type=Path, default=REPO_ROOT / LANG / "reasoning" / "splits" / "test.jsonl")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "report" / "phase3" / LANG)
    parser.add_argument("--n-examples", type=int, default=2)
    args = parser.parse_args()

    summary = run_comparison(
        pretrained_checkpoint=args.pretrained_checkpoint,
        finetuned_checkpoint=args.finetuned_checkpoint,
        tokenizer_path=args.tokenizer_path,
        test_split_path=args.test_split,
        sep_text=SEP_TEXT,
        lang=LANG,
        out_dir=args.out_dir,
        load_model_and_tokenizer_fn=load_model_and_tokenizer,
        capture_attention_fn=capture_attention,
        summarize_attention_fn=summarize_attention,
        plot_attention_heatmap_multihead_fn=plot_attention_heatmap_multihead,
        plot_entropy_by_layer_fn=plot_entropy_by_layer,
        plot_distance_by_layer_fn=plot_distance_by_layer,
        n_examples=args.n_examples,
    )
    print(f"[{LANG}] wrote attention comparison to {args.out_dir / 'attention_comparison.json'}")
    for stage in ("pretrained", "finetuned"):
        print(f"[{LANG}] {stage} avg entropy by layer: {summary['checkpoints'][stage]['avg_entropy_by_layer']}")


if __name__ == "__main__":
    main()
