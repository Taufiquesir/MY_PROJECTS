"""Parse a Kaggle training log (as printed by model/train_loop.py's train()) into
structured step/loss/lr/val series for plotting -- the raw evidence that training
actually converged, which the assignment explicitly requires alongside the final
checkpoint metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TRAIN_LINE = re.compile(
    r"step (\d+)/\d+ \| train_loss ([\d.]+) \| lr ([\d.e+-]+) \| \d+s elapsed"
)
_VAL_LINE = re.compile(r"step (\d+) \| val_loss ([\d.]+) \| val_ppl ([\d.]+)")


@dataclass
class TrainingCurve:
    train_steps: list[int]
    train_loss: list[float]
    lr: list[float]
    val_steps: list[int]
    val_loss: list[float]
    val_ppl: list[float]


def parse_training_log(log_path: Path) -> TrainingCurve:
    """Extract every (step, train_loss, lr) and (step, val_loss, val_ppl) line from
    a raw Kaggle log file. Lines that don't match either pattern (debugger
    warnings, dataset-path listings, etc., which the raw downloaded log is full
    of) are silently skipped -- this is meant to find the training signal inside a
    lot of unrelated Kaggle/Jupyter output, not validate the log's format.
    """
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")

    train_steps, train_loss, lr = [], [], []
    for m in _TRAIN_LINE.finditer(text):
        train_steps.append(int(m.group(1)))
        train_loss.append(float(m.group(2)))
        lr.append(float(m.group(3)))

    val_steps, val_loss, val_ppl = [], [], []
    for m in _VAL_LINE.finditer(text):
        val_steps.append(int(m.group(1)))
        val_loss.append(float(m.group(2)))
        val_ppl.append(float(m.group(3)))

    return TrainingCurve(
        train_steps=train_steps,
        train_loss=train_loss,
        lr=lr,
        val_steps=val_steps,
        val_loss=val_loss,
        val_ppl=val_ppl,
    )
