"""Perplexity (PPL) and bits-per-byte (BPB) evaluation for the ablation checkpoint.

Identical to ``hindi/eval/perplexity.py`` -- same PPL/BPB math, same reasoning for
why both are reported (see that module's docstring) -- the only difference is
``evaluate_perplexity`` here loads via ``bonus.model.checkpoint_ablation.
load_checkpoint`` (reconstructing ``DecoderOnlyTransformerNoPos``) instead of the
real model's loader. ``measure_bytes_per_token`` is reused directly from the real
module unchanged (re-imported here), since it only counts raw text bytes and
token-binary file size -- nothing about it depends on model architecture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from bonus.model.checkpoint_ablation import load_checkpoint
from hindi.model.data import TokenWindowDataset
from hindi.eval.perplexity import measure_bytes_per_token  # noqa: F401 -- re-exported, unchanged


@dataclass
class PerplexityResult:
    mean_nll: float
    ppl: float
    bpb: float
    n_tokens_evaluated: int


@torch.no_grad()
def evaluate_perplexity(
    checkpoint_path: Path,
    eval_bin: Path,
    bytes_per_token: float,
    device: str | None = None,
    batch_size: int = 32,
    max_batches: int | None = None,
) -> PerplexityResult:
    """Compute PPL and BPB for the ablation ``checkpoint_path`` over ``eval_bin``'s
    full token stream. See ``hindi.eval.perplexity.evaluate_perplexity`` for the
    full argument/behavior documentation -- identical here except for which
    checkpoint loader (and therefore which model class) is used.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, model_cfg, _ = load_checkpoint(checkpoint_path, device=device)
    model.to(device)
    model.eval()

    ds = TokenWindowDataset(eval_bin, model_cfg.max_seq_len)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)

    total_nll = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_nll += loss.item()
        total_tokens += y.numel()

    mean_nll = total_nll / total_tokens
    ppl = math.exp(mean_nll)
    bpb = (mean_nll / math.log(2)) / bytes_per_token

    return PerplexityResult(
        mean_nll=mean_nll,
        ppl=ppl,
        bpb=bpb,
        n_tokens_evaluated=total_tokens,
    )
