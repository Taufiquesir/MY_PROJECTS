"""Pretraining loop for the bonus no-positional-embedding ablation.

Reuses, unchanged, everything from ``hindi/model/train_loop.py`` that does not
name the real ``DecoderOnlyTransformer`` class directly: :class:`TrainConfig`,
``build_optimizer``, ``build_lr_scheduler``, and ``evaluate`` all only ever call
``model.named_parameters()`` / ``model(x)`` on whatever model they're given, so
they work identically for :class:`DecoderOnlyTransformerNoPos`. Also reuses
``TokenWindowDataset`` from ``hindi/model/data.py`` unchanged -- tokenizing and
windowing text has nothing to do with the model architecture.

Only the actual training loop is copied here, with the one line that constructs a
fresh model swapped for the ablation class -- every other line (data loading, the
RandomSampler-with-replacement memory fix, the training step, checkpointing
cadence) is identical to the graded pretraining loop, on purpose: this ablation
tests what changes when positional information is removed, not a different
training recipe.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hindi.model.data import TokenWindowDataset  # noqa: E402
from hindi.model.config import ModelConfig  # noqa: E402
from hindi.model.train_loop import TrainConfig, build_lr_scheduler, build_optimizer, evaluate  # noqa: E402

from .checkpoint_ablation import TrainingState, load_checkpoint, save_checkpoint  # noqa: E402
from .transformer_ablation import DecoderOnlyTransformerNoPos  # noqa: E402


def train(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    train_bin: Path,
    val_bin: Path,
    checkpoint_dir: Path,
    device: str | None = None,
    resume: bool = True,
) -> None:
    """Run (or resume) the ablation's pretraining, checkpointing to ``checkpoint_dir``.

    Same resumable-checkpoint contract as the graded pretraining run: weights +
    optimizer state + scheduler state + step + config, all in one file.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"

    train_ds = TokenWindowDataset(train_bin, model_cfg.max_seq_len)
    val_ds = TokenWindowDataset(val_bin, model_cfg.max_seq_len)
    train_sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=len(train_ds))
    val_sampler = torch.utils.data.RandomSampler(val_ds, replacement=True, num_samples=len(val_ds))
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, sampler=val_sampler, drop_last=True)

    step = 0
    best_val_loss = float("inf")

    if resume and latest_path.exists():
        print(f"[ablation] resuming from {latest_path}", flush=True)
        model, payload, loaded_cfg, state = load_checkpoint(latest_path, device=device)
        model.to(device)
        optimizer = build_optimizer(model, train_cfg)
        scheduler = build_lr_scheduler(optimizer, train_cfg)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        step = state.step
        best_val_loss = state.best_val_loss
        print(f"[ablation] resumed at step {step}, best_val_loss={best_val_loss:.4f}", flush=True)
    else:
        model = DecoderOnlyTransformerNoPos(**model_cfg.__dict__).to(device)
        optimizer = build_optimizer(model, train_cfg)
        scheduler = build_lr_scheduler(optimizer, train_cfg)
        print(f"[ablation] starting fresh: {model.num_parameters():,} trainable params", flush=True)

    model.train()
    train_iter = iter(train_loader)
    t0 = time.time()

    while step < train_cfg.max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % train_cfg.log_every == 0:
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            print(
                f"[ablation] step {step}/{train_cfg.max_steps} | train_loss {loss.item():.4f} | "
                f"lr {lr:.2e} | {elapsed:.0f}s elapsed",
                flush=True,
            )

        if step % train_cfg.eval_every == 0 or step == train_cfg.max_steps:
            val_loss = evaluate(model, val_loader, device, train_cfg.eval_batches)
            ppl = math.exp(min(val_loss, 20))
            print(f"[ablation] step {step} | val_loss {val_loss:.4f} | val_ppl {ppl:.2f}", flush=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    best_path, model, optimizer, scheduler, model_cfg,
                    TrainingState(step=step, best_val_loss=best_val_loss),
                )

        if step % train_cfg.checkpoint_every == 0 or step == train_cfg.max_steps:
            save_checkpoint(
                latest_path, model, optimizer, scheduler, model_cfg,
                TrainingState(step=step, best_val_loss=best_val_loss),
            )
            print(f"[ablation] checkpoint saved at step {step}", flush=True)

    print("[ablation] training complete", flush=True)
