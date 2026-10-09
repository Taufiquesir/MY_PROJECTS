"""Pretraining entry point for the bonus no-positional-embedding ablation
(Model H / Hindi).

Reuses ``prepare_binaries`` from ``hindi/train/train.py`` completely unchanged --
tokenizing the corpus has nothing to do with model architecture, so the ablation
reads the exact same tokenized binaries the standard Hindi model trains on (same
tokenizer, same train/val split, same everything except the model itself). Only
the model config and training loop differ, pointing at this folder's
``model_ablation.yaml`` / ``train_ablation.yaml`` and
``bonus.model.train_loop_ablation.train``.

Checkpoints land in ``bonus/checkpoints/`` by default -- kept fully separate from
``hindi/train/checkpoints/`` (the real model's) so neither run can ever overwrite
the other.

Usage (from the repo root)::

    python -m bonus.train.train_ablation
    python -m bonus.train.train_ablation --splits-dir /kaggle/input/lma-hindi-dataset/splits \\
        --checkpoint-dir /kaggle/working/checkpoints/hindi_ablation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hindi.model.config import ModelConfig  # noqa: E402
from hindi.model.train_loop import TrainConfig  # noqa: E402
from hindi.train.train import prepare_binaries  # noqa: E402

from bonus.model.train_loop_ablation import train  # noqa: E402

LANG = "hindi"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=REPO_ROOT / "data" / LANG / "splits")
    parser.add_argument("--tokenizer-path", type=Path, default=REPO_ROOT / LANG / "tokenizer" / f"{LANG}_tokenizer.model")
    parser.add_argument("--model-config", type=Path, default=REPO_ROOT / "bonus" / "configs" / "model_ablation.yaml")
    parser.add_argument("--train-config", type=Path, default=REPO_ROOT / "bonus" / "configs" / "train_ablation.yaml")
    parser.add_argument("--bin-dir", type=Path, default=REPO_ROOT / "data" / LANG / "tokenized")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "bonus" / "checkpoints")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    model_cfg = ModelConfig.load(args.model_config)
    train_cfg = TrainConfig.load(args.train_config)

    train_bin, val_bin = prepare_binaries(args.splits_dir, args.tokenizer_path, args.bin_dir)

    train(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        train_bin=train_bin,
        val_bin=val_bin,
        checkpoint_dir=args.checkpoint_dir,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
