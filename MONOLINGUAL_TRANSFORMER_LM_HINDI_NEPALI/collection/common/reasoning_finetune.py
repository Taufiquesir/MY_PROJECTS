"""Shared reasoning-finetuning loop: AdamW, linear-warmup + cosine-decay LR,
per-epoch validation, and checkpointing -- reused by both languages' finetune
scripts (``hindi/reasoning/finetune.py``, ``nepali/reasoning/finetune.py``), same as
``model/train_loop.py`` is shared by both languages' pretraining scripts.

Deliberately a *separate* loop from ``model/train_loop.py`` rather than a generalized
one, for reasons that matter to how each is read, not just implementation
convenience:

- Pretraining draws random fixed windows from an effectively unbounded token stream
  (RandomSampler with replacement, ``max_steps`` measured in gradient steps against
  a token budget). Finetuning has a small, fixed, enumerable dataset (~6,000
  examples) -- the natural unit is the epoch, not an arbitrary step budget, and
  every example should be seen a comparable number of times, so a plain shuffled
  DataLoader without replacement is used instead.
- Every pretraining loss term is real signal (every position predicts the next
  token of naturally occurring text). Finetuning's loss must additionally be masked
  down to just the answer tokens (see ``reasoning_data.ReasoningDataset``) --
  passing an all-ones mask like pretraining's would train the model to reproduce
  the given facts/question, not to answer them.
- Loading the *pretrained* checkpoint here uses only its ``model_state_dict`` (a
  fresh AdamW optimizer is constructed, not resumed from the pretraining run's
  optimizer state) -- finetuning is a new, differently-scaled optimization problem
  (small dataset, few epochs, typically a lower LR) rather than a continuation of
  the exact same pretraining trajectory, so warm-starting only the weights and not
  Adam's pretraining-scale moment estimates is the correct "start from that
  language's own pretrained checkpoint" per the assignment's Phase 3 protocol.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .reasoning_data import IGNORE_INDEX, ReasoningDataset


@dataclass
class FinetuneConfig:
    """Finetuning hyperparameters, separate from :class:`model.train_loop.TrainConfig`
    -- a lower peak LR and far fewer total steps than pretraining, since this is a
    short adaptation of an already-trained model on a small dataset, not a run
    starting from random initialization.
    """

    lr: float = 5e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 50
    epochs: int = 8
    batch_size: int = 16
    log_every: int = 20

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FinetuneConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def _masked_cross_entropy(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over only the positions where ``target_ids != IGNORE_INDEX``.

    ``F.cross_entropy``'s built-in ``ignore_index`` already excludes those positions
    from both the loss sum and its averaging denominator, so masked-out (prompt and
    padding) positions contribute no gradient and do not dilute the mean over the
    positions that do matter -- exactly the "loss only on the answer" requirement.
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1), ignore_index=IGNORE_INDEX
    )


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str) -> float:
    """Mean masked cross-entropy loss over a full pass of ``loader``.

    Unlike pretraining's ``evaluate`` (bounded to a fixed batch count, since a
    pretraining validation set can be huge), finetuning's val split is small enough
    that a full pass every check is cheap -- so this covers every example rather
    than a sample.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for input_ids, target_ids, _mask in loader:
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        logits = model(input_ids)
        loss = _masked_cross_entropy(logits, target_ids)
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(1, n_batches)


def build_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: FinetuneConfig, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay to 10% of peak LR -- same shape as
    pretraining's schedule (see ``model.train_loop.build_lr_scheduler``), just over
    a much shorter total step count (``total_steps`` = epochs * batches/epoch,
    known up front here since the dataset size is fixed, unlike pretraining's
    externally-set ``max_steps`` token budget).
    """

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return 0.1 + 0.9 * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_optimizer(model, cfg: FinetuneConfig) -> torch.optim.AdamW:
    """Same decay/no-decay parameter-group split as pretraining (see
    ``model.train_loop.build_optimizer``) -- weight decay on 2D+ weight matrices
    only, not on biases/LayerNorm/embedding scale parameters.
    """
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


def finetune(
    pretrained_checkpoint: Path,
    finetune_cfg: FinetuneConfig,
    train_dataset: ReasoningDataset,
    val_dataset: ReasoningDataset,
    checkpoint_dir: Path,
    load_checkpoint: Callable,
    save_checkpoint: Callable,
    training_state_cls: Callable,
    device: str | None = None,
    resume: bool = True,
) -> dict:
    """Finetune a pretrained checkpoint on ``train_dataset``, validating every epoch.

    ``load_checkpoint``/``save_checkpoint``/``training_state_cls`` are passed in by
    the caller rather than imported here, deliberately: this module is shared by
    both languages' finetune scripts (``hindi/reasoning/finetune.py``,
    ``nepali/reasoning/finetune.py``), and Model H / Model L must never share code
    that could couple one language's finetuning run to the other's package -- even
    though ``hindi/model/checkpoint.py`` and ``nepali/model/checkpoint.py`` happen to
    be identical files today, each language's finetune script imports and passes its
    *own* copy, so the two runs stay independently swappable (e.g. if one language's
    architecture code diverges later, only that language's caller needs to change).

    Resumable the same way pretraining is: if ``checkpoint_dir/latest.pt`` already
    exists and ``resume=True``, training resumes from its saved epoch/optimizer/
    scheduler state instead of restarting from ``pretrained_checkpoint`` -- the
    assignment requires finetuned checkpoints be saved "in the same resume-capable
    format as pretraining", and this reuses the identical
    ``model/checkpoint.py::save_checkpoint``/``load_checkpoint`` functions (weights +
    optimizer + scheduler + step + config all in one file) rather than a parallel
    format.

    Returns a dict of per-epoch train/val loss, for the report's finetuning curve.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"

    train_loader = DataLoader(train_dataset, batch_size=finetune_cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=finetune_cfg.batch_size, shuffle=False)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * finetune_cfg.epochs

    history: dict = {"train_loss": [], "val_loss": []}
    start_epoch = 0
    best_val_loss = float("inf")

    if resume and latest_path.exists():
        print(f"resuming finetune from {latest_path}", flush=True)
        model, payload, model_cfg, state = load_checkpoint(latest_path, device=device)
        model.to(device)
        optimizer = build_optimizer(model, finetune_cfg)
        scheduler = build_lr_scheduler(optimizer, finetune_cfg, total_steps)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = state.step  # step field doubles as "epochs completed" here
        best_val_loss = state.best_val_loss
        history = payload.get("history", history)
        print(f"resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}", flush=True)
    else:
        print(f"starting finetune from pretrained checkpoint {pretrained_checkpoint}", flush=True)
        model, _payload, model_cfg, _state = load_checkpoint(pretrained_checkpoint, device=device)
        model.to(device)
        optimizer = build_optimizer(model, finetune_cfg)
        scheduler = build_lr_scheduler(optimizer, finetune_cfg, total_steps)

    model.train()
    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, finetune_cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for input_ids, target_ids, _mask in train_loader:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            logits = model(input_ids)
            loss = _masked_cross_entropy(logits, target_ids)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), finetune_cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss += loss.item()
            n_batches += 1

            if global_step % finetune_cfg.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"epoch {epoch + 1}/{finetune_cfg.epochs} step {global_step} | train_loss {loss.item():.4f} | lr {lr:.2e}", flush=True)

        train_loss = epoch_loss / max(1, n_batches)
        val_loss = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch + 1} complete | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}", flush=True)

        state = training_state_cls(step=epoch + 1, best_val_loss=min(best_val_loss, val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_path, model, optimizer, scheduler, model_cfg, state)
        save_checkpoint(latest_path, model, optimizer, scheduler, model_cfg, state)
        # history is not part of save_checkpoint's fixed schema (weights/optimizer/
        # scheduler/step/config) -- appended into the saved file directly here so a
        # resumed run can keep extending the same curve the report plots from.
        payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        payload["history"] = history
        torch.save(payload, latest_path)

    print("finetuning complete", flush=True)
    return history
