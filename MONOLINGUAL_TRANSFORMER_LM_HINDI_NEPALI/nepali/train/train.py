"""Pretraining entry point for Model L (Nepali).

Prepares the tokenized binary streams (once; skipped on later runs if they already
exist) and runs :func:`model.train_loop.train`.

Usage (from the repo root, so the default relative paths below resolve correctly)::

    python -m nepali.train.train
    python -m nepali.train.train --splits-dir /kaggle/input/lma-nepali-dataset/splits \\
        --checkpoint-dir /kaggle/working/checkpoints/nepali

``--splits-dir`` defaults to ``data/nepali/splits`` (this repo's local layout, where
the language name is a path segment under a shared ``data/`` root). The Kaggle
mirror of this corpus is a *per-language* dataset (``lma-nepali-dataset`` holds only
Nepali), so ``splits/`` sits directly at that dataset's root with no ``nepali/``
segment in between -- hence a separate flag rather than reusing ``--data-root`` with
an assumed ``<data_root>/nepali/splits`` shape, which would be wrong on Kaggle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nepali.model.config import ModelConfig  # noqa: E402
from nepali.model.data import tokenize_split_to_binary  # noqa: E402
from nepali.model.train_loop import TrainConfig, train  # noqa: E402

LANG = "nepali"


def prepare_binaries(splits_dir: Path, tokenizer_path: Path, bin_dir: Path) -> tuple[Path, Path]:
    """Tokenize train/val splits to flat uint16 binaries, if not already done.

    ``splits_dir`` is expected to directly contain ``train/`` and ``val/``
    subdirectories of JSONL shards (i.e. this is already the language-specific
    ``splits/`` directory, not a multi-language root) -- matches
    ``hindi/train/train.py``'s signature, see its docstring for why that path is
    passed in explicitly rather than assumed. A caller passing a data root plus
    an implicit ``<root>/nepali/splits`` shape (this function's old signature)
    would silently tokenize an empty/nonexistent directory, since ``iter_records``
    swallows a missing directory the same as an empty one.

    Idempotent: an existing binary is trusted as-is and not regenerated, so
    re-running this script after an interruption does not repeat the (relatively
    slow, single-threaded) tokenization pass.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    train_bin = bin_dir / "train.bin"
    val_bin = bin_dir / "val.bin"
    # A 0-byte leftover from an earlier failed/interrupted tokenization pass (e.g. a
    # source directory that was empty or misconfigured at the time) satisfies
    # .exists() and would otherwise be silently reused forever -- checking st_size
    # catches that case and re-tokenizes instead of handing an empty file to training.
    if not train_bin.exists() or train_bin.stat().st_size == 0:
        print(f"[{LANG}] tokenizing train split -> {train_bin}", flush=True)
        n = tokenize_split_to_binary(splits_dir / "train", tokenizer_path, train_bin)
        print(f"[{LANG}] train: {n:,} tokens", flush=True)
        if n == 0:
            raise ValueError(
                f"[{LANG}] tokenized 0 tokens from {splits_dir / 'train'} -- check "
                "that this directory actually contains shard-*.jsonl(.gz) files "
                "with a 'text' field before re-running."
            )
    if not val_bin.exists() or val_bin.stat().st_size == 0:
        print(f"[{LANG}] tokenizing val split -> {val_bin}", flush=True)
        n = tokenize_split_to_binary(splits_dir / "val", tokenizer_path, val_bin)
        print(f"[{LANG}] val: {n:,} tokens", flush=True)
        if n == 0:
            raise ValueError(
                f"[{LANG}] tokenized 0 tokens from {splits_dir / 'val'} -- check "
                "that this directory actually contains shard-*.jsonl(.gz) files "
                "with a 'text' field before re-running."
            )
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
