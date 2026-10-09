"""Checkpoint save/resume round-trip test -- the assignment's mandatory requirement
that a training run can resume with model weights, optimizer state, scheduler state,
and step count all intact.
"""

import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from hindi.model.checkpoint import TrainingState, load_checkpoint, resume_optimizer_and_scheduler, save_checkpoint
from hindi.model.config import ModelConfig
from hindi.model.transformer import DecoderOnlyTransformer


def _train_a_few_steps(model, optimizer, scheduler, vocab_size, n_steps=3):
    loss = None
    for _ in range(n_steps):
        x = torch.randint(0, vocab_size, (2, 6))
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), x[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return loss


def test_checkpoint_round_trip_preserves_everything():
    cfg = ModelConfig(vocab_size=50, d_model=16, n_layers=2, n_heads=2, d_ff=32, max_seq_len=8, dropout=0.0)
    model = DecoderOnlyTransformer(**cfg.__dict__)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    final_loss = _train_a_few_steps(model, optimizer, scheduler, cfg.vocab_size)

    with tempfile.TemporaryDirectory() as d:
        ckpt_path = Path(d) / "ckpt.pt"
        save_checkpoint(ckpt_path, model, optimizer, scheduler, cfg, TrainingState(step=3, best_val_loss=final_loss.item()))
        assert ckpt_path.exists()

        model2, payload, cfg2, state2 = load_checkpoint(ckpt_path)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=0.9)
        resume_optimizer_and_scheduler(payload, optimizer2, scheduler2)

    assert state2.step == 3
    assert cfg2 == cfg

    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.equal(p1, p2), f"weight mismatch in {n1}"

    sd1, sd2 = optimizer.state_dict()["state"], optimizer2.state_dict()["state"]
    assert len(sd1) == len(sd2) > 0

    assert scheduler.get_last_lr() == scheduler2.get_last_lr()


def test_resumed_model_produces_identical_output():
    """Beyond matching tensors, a resumed model must actually behave identically."""
    cfg = ModelConfig(vocab_size=50, d_model=16, n_layers=2, n_heads=2, d_ff=32, max_seq_len=8, dropout=0.0)
    model = DecoderOnlyTransformer(**cfg.__dict__)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    _train_a_few_steps(model, optimizer, scheduler, cfg.vocab_size)
    model.eval()

    x = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        logits_before = model(x)

    with tempfile.TemporaryDirectory() as d:
        ckpt_path = Path(d) / "ckpt.pt"
        save_checkpoint(ckpt_path, model, optimizer, scheduler, cfg, TrainingState(step=3, best_val_loss=0.0))
        model2, _, _, _ = load_checkpoint(ckpt_path)

    model2.eval()
    with torch.no_grad():
        logits_after = model2(x)

    assert torch.equal(logits_before, logits_after)
