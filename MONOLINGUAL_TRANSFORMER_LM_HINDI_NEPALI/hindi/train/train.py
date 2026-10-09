"""Pretraining entry point for Model H (Hindi).

Prepares the tokenized binary streams (once; skipped on later runs if they already
exist) and runs :func:`model.train_loop.train`.

Usage (from the repo root, so the default relative paths below resolve correctly)::

    python -m hindi.train.train
    python -m hindi.train.train --splits-dir /kaggle/input/lma-hindi-dataset/splits \\
        --checkpoint-dir /kaggle/working/checkpoints/hindi

``--splits-dir`` defaults to ``data/hindi/splits`` (this repo's local layout, where
the language name is a path segment under a shared ``data/`` root). The Kaggle
mirror of this corpus is a *per-language* dataset (``lma-hindi-dataset`` holds only
Hindi), so ``splits/`` sits directly at that dataset's root with no ``hindi/``
segment in between -- hence a separate flag rather than reusing ``--data-root`` with
an assumed ``<data_root>/hindi/splits`` shape, which would be wrong on Kaggle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hindi.model.config import ModelConfig  # noqa: E402
from hindi.model.data import tokenize_split_to_binary  # noqa: E402
from hindi.model.train_loop import TrainConfig, train  # noqa: E402

LANG = "hindi"


def prepare_binaries(splits_dir: Path, tokenizer_path: Path, bin_dir: Path) -> tuple[Path, Path]:
    """Tokenize train/val splits to flat uint16 binaries, if not already done.

    ``splits_dir`` is expected to directly contain ``train/`` and ``val/``
    subdirectories of JSONL shards (i.e. this is already the language-specific
    ``splits/`` directory, not a multi-language root) -- see this module's
    docstring for why that path is passed in explicitly rather than assumed.

    Idempotent: an existing binary is trusted as-is and not regenerated, so
    re-running this script after an interruption does not repeat the (relatively
    slow, single-threaded) tokenization pass.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    train_bin = bin_dir / "train.bin"
    val_bin = bin_dir / "val.bin"
    if not train_bin.exists():
        print(f"[{LANG}] tokenizing train split -> {train_bin}", flush=True)
        n = tokenize_split_to_binary(splits_dir / "train", tokenizer_path, train_bin)
        print(f"[{LANG}] train: {n:,} tokens", flush=True)
    if not val_bin.exists():
        print(f"[{LANG}] tokenizing val split -> {val_bin}", flush=True)
        n = tokenize_split_to_binary(splits_dir / "val", tokenizer_path, val_bin)
        print(f"[{LANG}] val: {n:,} tokens", flush=True)
    return train_bin, val_bin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=REPO_ROOT / "data" / LANG / "splits")
    parser.add_argument("--tokenizer-path", type=Path, default=REPO_ROOT / LANG / "tokenizer" / f"{LANG}_tokenizer.model")
    parser.add_argument("--model-config", type=Path, default=REPO_ROOT / LANG / "configs" / "model.yaml")
    parser.add_argument("--train-config", type=Path, default=REPO_ROOT / LANG / "configs" / "train.yaml")
    parser.add_argument("--bin-dir", type=Path, default=REPO_ROOT / "data" / LANG / "tokenized")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / LANG / "train" / "checkpoints")
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
