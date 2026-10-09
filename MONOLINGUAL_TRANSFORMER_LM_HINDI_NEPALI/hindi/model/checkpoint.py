"""Checkpoint save/resume.

The assignment makes this a hard requirement: a training run must be able to resume
after an interruption (e.g. a Colab/Kaggle session timing out) with no loss of
progress. A single checkpoint file bundles everything needed to reconstruct the
*exact* training state, not just the model weights:

- model weights (``state_dict``)
- optimizer state (``state_dict`` -- Adam's per-parameter moment estimates; resuming
  without these is not equivalent to resuming, since Adam's effective per-parameter
  learning rate depends on its running moment estimates, not just the raw LR)
- LR scheduler state (``state_dict`` -- so a warmup/decay schedule continues from the
  correct point rather than restarting)
- the training step counter (so logging, the LR schedule, and "how much more to
  train" all stay consistent across a resume)
- the model config (so a checkpoint is self-describing: it can be loaded into a
  freshly-constructed model without the caller needing to already know the exact
  architecture hyperparameters used to produce it)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig
from .transformer import DecoderOnlyTransformer


@dataclass
class TrainingState:
    """Everything needed to resume a training run from where it left off."""

    step: int
    best_val_loss: float


def save_checkpoint(
    path: Path,
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: ModelConfig,
    state: TrainingState,
) -> None:
    """Write one checkpoint file containing model + optimizer + scheduler + step + config.

    Written to a temporary path first and then renamed into place (``os.replace`` is
    atomic on the same filesystem) so a crash mid-write can never leave a corrupt,
    half-written file at ``path`` -- a resume always sees either the previous good
    checkpoint or a complete new one, never a torn one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config.__dict__,
        "step": state.step,
        "best_val_loss": state.best_val_loss,
    }
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(
    path: Path,
    device: str = "cpu",
) -> tuple[DecoderOnlyTransformer, dict, ModelConfig, TrainingState]:
    """Load a checkpoint, reconstructing the model from its embedded config.

    Returns the model (weights loaded, not yet moved to ``device`` by the optimizer
    -- caller does that), the raw optimizer state dict (caller constructs the actual
    optimizer/scheduler objects first, since their constructor arguments like LR
    aren't stored here, then calls ``.load_state_dict`` on them), the config, and the
    training state (step, best_val_loss) to resume from.
    """
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = ModelConfig(**payload["config"])
    model = DecoderOnlyTransformer(**config.__dict__)
    model.load_state_dict(payload["model_state_dict"])
    state = TrainingState(step=payload["step"], best_val_loss=payload["best_val_loss"])
    return model, payload, config, state


def resume_optimizer_and_scheduler(
    payload: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    """Restore optimizer/scheduler state from a checkpoint payload returned by
    :func:`load_checkpoint`. Split out from ``load_checkpoint`` itself because the
    optimizer/scheduler must already exist (constructed against the resumed model's
    parameters) before their state can be loaded into them.
    """
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
