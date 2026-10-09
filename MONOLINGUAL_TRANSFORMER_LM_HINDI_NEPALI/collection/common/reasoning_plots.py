"""Plotting for the Phase 3 finetuning curve -- kept separate from computation
(the actual finetuning loop in ``reasoning_finetune.py``) per this project's
established convention of never mixing computation and visualization in one
function (see ``collection/common/plots.py`` from Phase 1 and
``hindi/eval/plots.py`` from Phase 2).

A dedicated function rather than reusing ``eval/plots.py``'s
``plot_training_curve`` because the x-axis is a different unit: pretraining logs
loss against *step* (with train/val logged at different intervals), while
finetuning here logs one train/val loss pair per *epoch* (the whole point of
switching units for Phase 3 -- see ``reasoning_finetune.py``'s module docstring).
Reusing the step-based plot would mean pretending a fixed batches-per-epoch count
is a step number, which is technically true but a confusing axis label for a
reader trying to compare this plot to Phase 2's.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_finetune_loss_curve(train_loss: list[float], val_loss: list[float], lang: str, out_path: Path) -> None:
    """Train/val loss per epoch during reasoning finetuning."""
    epochs = list(range(1, len(train_loss) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, marker="o", color="tab:blue", label="Train loss")
    ax.plot(epochs, val_loss, marker="o", color="tab:orange", label="Validation loss")
    ax.set_title(f"{lang.capitalize()} -- Reasoning Finetuning Loss by Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked cross-entropy loss (answer tokens only)")
    ax.set_xticks(epochs)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
