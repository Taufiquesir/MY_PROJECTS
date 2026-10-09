"""Pretrained-vs-finetuned reasoning accuracy for Model H (Hindi).

Compares ``hindi/train/checkpoints/best.pt`` (Phase 2 pretrained) against
``hindi/reasoning/checkpoints/best.pt`` (Phase 3 finetuned) on
``hindi/reasoning/splits/test.jsonl`` -- the held-out entities/attribute the
finetuning run never trained on (see ``hindi/reasoning/generate.py``).

Usage (from the repo root)::

    python -m hindi.reasoning.eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from collection.common.reasoning_eval import compare_pretrained_vs_finetuned, write_results  # noqa: E402
from hindi.model.checkpoint import load_checkpoint  # noqa: E402

LANG = "hindi"
SEP_TEXT = "उत्तर:"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=REPO_ROOT / LANG / "train" / "checkpoints" / "best.pt")
    parser.add_argument("--finetuned-checkpoint", type=Path, default=REPO_ROOT / LANG / "reasoning" / "checkpoints" / "best.pt")
    parser.add_argument("--tokenizer-path", type=Path, default=REPO_ROOT / LANG / "tokenizer" / f"{LANG}_tokenizer.model")
    parser.add_argument("--test-split", type=Path, default=REPO_ROOT / LANG / "reasoning" / "splits" / "test.jsonl")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "report" / "phase3" / LANG / "reasoning_metrics.json")
    args = parser.parse_args()

    results = compare_pretrained_vs_finetuned(
        pretrained_checkpoint=args.pretrained_checkpoint,
        finetuned_checkpoint=args.finetuned_checkpoint,
        tokenizer_path=args.tokenizer_path,
        test_split_path=args.test_split,
        sep_text=SEP_TEXT,
        load_checkpoint_fn=load_checkpoint,
    )
    write_results(results, args.out)

    for label in ("pretrained", "finetuned"):
        r = results[label]
        print(f"[{LANG}] {label}: accuracy={r['accuracy']:.4f} ({r['n_correct']}/{r['n_examples']})")
        print(f"[{LANG}] {label} by kind: {json.dumps(r['accuracy_by_kind'], ensure_ascii=False)}")
    print(f"[{LANG}] wrote results to {args.out}")


if __name__ == "__main__":
    main()
