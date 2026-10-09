"""Shared pretraining loop: AdamW, linear-warmup + cosine-decay LR schedule,
periodic validation, and checkpointing. One copy used by both languages' training
scripts (``hindi/train/train.py``, ``nepali/train/train.py``) -- the loop logic is
identical, only the data/tokenizer/output paths differ per language.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .checkpoint import TrainingState, load_checkpoint, resume_optimizer_and_scheduler, save_checkpoint
from .config import ModelConfig
from .data import TokenWindowDataset
from .transformer import DecoderOnlyTransformer


@dataclass
class TrainConfig:
    """Training hyperparameters -- separate from :class:`model.config.ModelConfig`
    (architecture) since these can change between runs without changing the model
    the checkpoint describes."""

    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    max_steps: int = 50_000
    batch_size: int = 32
    eval_every: int = 500
    eval_batches: int = 50
    checkpoint_every: int = 500
    log_every: int = 50

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TrainConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def build_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup for ``warmup_steps``, then cosine decay to 10% of peak LR over
    the remaining steps to ``max_steps``. Warmup avoids a large, poorly-conditioned
    update at step 0 when Adam's moment estimates are still all zero; cosine decay
    (rather than a fixed LR or a step schedule) gives a smooth reduction that in
    practice tends to land in a better final loss basin than either alternative at
    this model scale, without needing to hand-tune decay milestones.
    """

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return 0.1 + 0.9 * cosine  # decay to 10% of peak, never fully to 0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_optimizer(model: DecoderOnlyTransformer, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay applied only to 2D+ weight matrices, not to biases or
    LayerNorm/embedding scale parameters -- decaying those regularizes a quantity
    (a shift or a norm scale) that isn't the kind of large weight matrix weight
    decay is meant to shrink, and in practice decaying LayerNorm/bias terms tends to
    hurt more than it helps at this scale.
    """
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    param_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


@torch.no_grad()
def evaluate(model: DecoderOnlyTransformer, val_loader: DataLoader, device: str, max_batches: int) -> float:
    """Mean cross-entropy loss over up to ``max_batches`` batches of ``val_loader``.

    Bounded to a fixed number of batches rather than the full validation set every
    call, since this runs periodically *during* training (not just once at the end)
    and the full val set can be large -- a bounded, but still substantial, sample is
    enough to track whether validation loss is improving without spending as much
    wall-clock per check as a full pass.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(1, n_batches)


def train(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    train_bin: Path,
    val_bin: Path,
    checkpoint_dir: Path,
    device: str | None = None,
    resume: bool = True,
) -> None:
    """Run (or resume) pretraining, checkpointing periodically to ``checkpoint_dir``.

    Resumable by construction: if ``checkpoint_dir/latest.pt`` exists and
    ``resume=True``, training picks up from its saved step/optimizer/scheduler state
    rather than starting over -- this is what makes an interrupted Kaggle/Colab
    session recoverable, per the assignment's mandatory checkpoint-resume requirement.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"

    train_ds = TokenWindowDataset(train_bin, model_cfg.max_seq_len)
    val_ds = TokenWindowDataset(val_bin, model_cfg.max_seq_len)
    # shuffle=True would make DataLoader build a RandomSampler, which calls
    # torch.randperm(len(dataset)) -- a full permutation held as one int64 tensor in
    # host RAM. len(train_ds) is ~683 million (one entry per possible window start
    # offset in the token stream), so that permutation alone is ~5.2GB, on top of
    # everything else already resident -- this is what was OOM-killing the Kaggle
    # session right after model/optimizer construction. A RandomSampler with
    # replacement=True sample random start offsets independently instead, which is
    # statistically equivalent for this purpose (each training window is still drawn
    # uniformly at random) but needs no permutation array at all.
    train_sampler = torch.utils.data.RandomSampler(train_ds, replacement=True, num_samples=len(train_ds))
    val_sampler = torch.utils.data.RandomSampler(val_ds, replacement=True, num_samples=len(val_ds))
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, sampler=val_sampler, drop_last=True)

    step = 0
    best_val_loss = float("inf")

    if resume and latest_path.exists():
        print(f"resuming from {latest_path}", flush=True)
        model, payload, loaded_cfg, state = load_checkpoint(latest_path, device=device)
        model.to(device)
        optimizer = build_optimizer(model, train_cfg)
        scheduler = build_lr_scheduler(optimizer, train_cfg)
        resume_optimizer_and_scheduler(payload, optimizer, scheduler)
        step = state.step
        best_val_loss = state.best_val_loss
        print(f"resumed at step {step}, best_val_loss={best_val_loss:.4f}", flush=True)
    else:
        model = DecoderOnlyTransformer(**model_cfg.__dict__).to(device)
        optimizer = build_optimizer(model, train_cfg)
        scheduler = build_lr_scheduler(optimizer, train_cfg)
        print(f"starting fresh: {model.num_parameters():,} trainable params", flush=True)

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
                f"step {step}/{train_cfg.max_steps} | train_loss {loss.item():.4f} | "
                f"lr {lr:.2e} | {elapsed:.0f}s elapsed",
                flush=True,
            )

        if step % train_cfg.eval_every == 0 or step == train_cfg.max_steps:
            val_loss = evaluate(model, val_loader, device, train_cfg.eval_batches)
            ppl = math.exp(min(val_loss, 20))  # cap to avoid overflow on an early, bad checkpoint
            print(f"step {step} | val_loss {val_loss:.4f} | val_ppl {ppl:.2f}", flush=True)
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
            print(f"checkpoint saved at step {step}", flush=True)

    print("training complete", flush=True)
