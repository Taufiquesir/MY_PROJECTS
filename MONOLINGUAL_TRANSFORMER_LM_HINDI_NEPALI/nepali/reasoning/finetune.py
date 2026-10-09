"""Reasoning-finetuning entry point for Model L (Nepali).

Starts from Model L's own Phase 2 pretrained checkpoint (``nepali/train/checkpoints/
best.pt``), keeps the Nepali tokenizer/vocabulary completely fixed, and finetunes on
``nepali/reasoning/splits/`` -- the synthetic dataset from ``nepali/reasoning/
generate.py``. Never touches Hindi data, checkpoints, or code. See
``hindi/reasoning/finetune.py`` for the mirrored Hindi entry point.

Usage (from the repo root)::

    python -m nepali.reasoning.finetune
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import sentencepiece as spm  # noqa: E402

from collection.common.reasoning_data import ReasoningDataset, load_examples  # noqa: E402
from collection.common.reasoning_finetune import FinetuneConfig, finetune  # noqa: E402
from collection.common.reasoning_plots import plot_finetune_loss_curve  # noqa: E402
from nepali.model.checkpoint import TrainingState, load_checkpoint, save_checkpoint  # noqa: E402

LANG = "nepali"
SEP_TEXT = "जवाफ:"  # "answer:" in Nepali -- see reasoning_data.ReasoningDataset docstring


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=REPO_ROOT / LANG / "reasoning" / "splits")
    parser.add_argument("--tokenizer-path", type=Path, default=REPO_ROOT / LANG / "tokenizer" / f"{LANG}_tokenizer.model")
    parser.add_argument("--pretrained-checkpoint", type=Path, default=REPO_ROOT / LANG / "train" / "checkpoints" / "best.pt")
    parser.add_argument("--finetune-config", type=Path, default=REPO_ROOT / LANG / "configs" / "finetune.yaml")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / LANG / "reasoning" / "checkpoints")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--plot-out", type=Path, default=REPO_ROOT / "report" / "phase3" / LANG / "plots" / "finetune_loss_curve.png")
    args = parser.parse_args()

    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer_path))

    if args.finetune_config.exists():
        ft_cfg = FinetuneConfig.load(args.finetune_config)
    else:
        ft_cfg = FinetuneConfig()
        ft_cfg.save(args.finetune_config)
        print(f"[{LANG}] wrote default finetune config to {args.finetune_config}", flush=True)

    train_examples = load_examples(args.splits_dir / "train.jsonl")
    val_examples = load_examples(args.splits_dir / "val.jsonl")
    train_ds = ReasoningDataset(train_examples, tokenizer, args.max_seq_len, SEP_TEXT)
    val_ds = ReasoningDataset(val_examples, tokenizer, args.max_seq_len, SEP_TEXT)
    print(f"[{LANG}] train examples: {len(train_ds)} ({train_ds.n_dropped} dropped as over-length)", flush=True)
    print(f"[{LANG}] val examples: {len(val_ds)} ({val_ds.n_dropped} dropped as over-length)", flush=True)

    history = finetune(
        pretrained_checkpoint=args.pretrained_checkpoint,
        finetune_cfg=ft_cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        checkpoint_dir=args.checkpoint_dir,
        load_checkpoint=load_checkpoint,
        save_checkpoint=save_checkpoint,
        training_state_cls=TrainingState,
        resume=not args.no_resume,
    )
    plot_finetune_loss_curve(history["train_loss"], history["val_loss"], LANG, args.plot_out)
    print(f"[{LANG}] wrote finetuning loss curve to {args.plot_out}", flush=True)


if __name__ == "__main__":
    main()
