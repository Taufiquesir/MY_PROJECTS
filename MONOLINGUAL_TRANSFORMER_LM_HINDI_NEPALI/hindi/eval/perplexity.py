"""Perplexity (PPL) and bits-per-byte (BPB) evaluation for a trained checkpoint.

Both are computed from the same underlying quantity -- mean cross-entropy loss over
a held-out token stream -- but express it in two different, commonly-reported units:

- **PPL** = ``exp(mean_nll)``, in units of "effective branching factor" over the
  tokenizer's own vocabulary. Directly comparable across checkpoints of the *same*
  tokenizer, but NOT across Hindi vs Nepali (or any two models with different
  vocabularies/tokenizers), since a token means a different amount of text in each.
- **BPB** = ``mean_nll_in_bits * (tokens / bytes)``, i.e. average bits needed per
  *byte* of raw UTF-8 text rather than per token. This normalizes away tokenizer
  differences (vocab size, average tokens-per-character), which is exactly why the
  assignment wants it alongside PPL: it is the metric that IS comparable between the
  two independently-tokenized languages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from ..model.checkpoint import load_checkpoint
from ..model.data import TokenWindowDataset


@dataclass
class PerplexityResult:
    mean_nll: float  # mean negative log-likelihood, nats/token
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
    """Compute PPL and BPB for ``checkpoint_path`` over ``eval_bin``'s full token stream.

    Args:
        checkpoint_path: a ``.pt`` file written by :func:`model.checkpoint.save_checkpoint`.
        eval_bin: a flat uint16 token-id binary written by
            :func:`model.data.tokenize_split_to_binary` (typically the test split,
            held out from both training and validation-during-training).
        bytes_per_token: measured ``raw_utf8_bytes / n_tokens`` for this split, used
            to convert nats/token into bits/byte. Must be measured from the actual
            split being evaluated, not assumed, since BPB's whole point is to be a
            tokenizer-agnostic comparison -- an assumed ratio would defeat that.
        max_batches: caps evaluation to this many batches if set, for a quick sanity
            check; ``None`` evaluates the entire split (the real reported number).
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
        # sum (not mean) so token counts across batches combine correctly even if
        # a later change makes batches unequal size.
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_nll += loss.item()
        total_tokens += y.numel()

    mean_nll = total_nll / total_tokens  # nats/token
    ppl = math.exp(mean_nll)
    # nats -> bits: divide by ln(2). Then nats(bits)/token -> bits/byte via the
    # measured tokens-per-byte ratio for this split.
    bpb = (mean_nll / math.log(2)) / bytes_per_token

    return PerplexityResult(
        mean_nll=mean_nll,
        ppl=ppl,
        bpb=bpb,
        n_tokens_evaluated=total_tokens,
    )


def measure_bytes_per_token(bin_path: Path, split_dir: Path, tokenizer_path: Path) -> float:
    """Measure ``raw_utf8_bytes / n_tokens`` for one split, from the actual source
    text and its tokenized binary -- the ratio BPB needs, computed rather than
    assumed (see :func:`evaluate_perplexity`'s docstring for why that matters).
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from collection.common.record import iter_records

    total_bytes = sum(len(rec.text.encode("utf-8")) for rec in iter_records(split_dir))
    n_tokens = Path(bin_path).stat().st_size // 2  # uint16 = 2 bytes/token
    return total_bytes / n_tokens
