"""Checkpoint save/resume for the bonus no-positional-embedding ablation.

Same resumable format as the graded pretraining checkpoints (weights + optimizer
state + scheduler state + step + config, all in one file, atomic tmp-then-rename
write) -- the assignment requires the ablation to use the same resume-capable
format as the main run, so this mirrors ``hindi/model/checkpoint.py`` exactly,
just reconstructing :class:`DecoderOnlyTransformerNoPos` instead of the real
model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from hindi.model.config import ModelConfig
from .transformer_ablation import DecoderOnlyTransformerNoPos


@dataclass
class TrainingState:
    """Everything needed to resume a training run from where it left off."""

    step: int
    best_val_loss: float


def save_checkpoint(
    path: Path,
    model: DecoderOnlyTransformerNoPos,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: ModelConfig,
    state: TrainingState,
) -> None:
    """Write one checkpoint file containing model + optimizer + scheduler + step + config.

    Written to a temporary path first and then renamed into place, same
    crash-safety reasoning as the graded pretraining checkpoint code.
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
) -> tuple[DecoderOnlyTransformerNoPos, dict, ModelConfig, TrainingState]:
    """Load an ablation checkpoint, reconstructing a ``DecoderOnlyTransformerNoPos``
    from its embedded config. Returns the model (weights loaded, not yet moved to
    ``device``), the raw payload (caller uses it to restore optimizer/scheduler
    state), the config, and the training state to resume from.
    """
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = ModelConfig(**payload["config"])
    model = DecoderOnlyTransformerNoPos(**config.__dict__)
    model.load_state_dict(payload["model_state_dict"])
    state = TrainingState(step=payload["step"], best_val_loss=payload["best_val_loss"])
    return model, payload, config, state


def resume_optimizer_and_scheduler(
    payload: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    """Restore optimizer/scheduler state from a checkpoint payload returned by
    :func:`load_checkpoint`."""
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
