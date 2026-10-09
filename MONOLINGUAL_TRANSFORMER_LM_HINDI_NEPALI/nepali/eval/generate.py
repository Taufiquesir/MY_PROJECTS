"""Autoregressive text generation from a trained checkpoint.

Shared by both the BLEU/chrF/ROUGE-L generation-quality metrics and the diversity
diagnostics (repetition rate, Distinct-n) in ``eval/quality.py`` -- both need the
same "feed a prompt, produce a continuation" primitive, just scored differently
afterward.

Two decoding strategies, per the assignment's explicit generation-quality protocol
(evaluate both, not just one):

- **Greedy**: always take the highest-probability next token. Deterministic --
  the same prompt always produces the same continuation. Included as the
  no-randomness baseline the temperature-sampled runs are compared against.
- **Temperature sampling** at temperatures 0.5, 1.0, and 1.5: scale logits by
  ``1/temperature`` before softmax, then sample from the resulting distribution.
  temperature<1 sharpens the distribution toward the model's most-confident
  tokens (closer to greedy but not identical, since it still samples rather than
  argmaxes); temperature=1.0 samples from the model's distribution exactly as
  trained; temperature>1 flattens the distribution, increasing randomness. No
  top-k/top-p truncation, so all three settings differ *only* in temperature --
  isolating that one variable is the point of reporting all three.
"""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

from ..model.checkpoint import load_checkpoint
from ..model.transformer import DecoderOnlyTransformer


@torch.no_grad()
def generate_greedy(
    model: DecoderOnlyTransformer,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    max_new_tokens: int,
    device: str,
    eos_id: int | None = None,
) -> str:
    """Deterministic argmax decoding: always emit the single highest-probability
    next token. See module docstring for why this is reported alongside
    temperature sampling rather than instead of it.
    """
    model.eval()
    ids = tokenizer.encode(prompt, out_type=int)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = input_ids[:, -model.max_seq_len :]
        logits = model(context)  # (1, T, vocab_size)
        next_id = torch.argmax(logits[0, -1, :]).view(1, 1)

        if eos_id is not None and next_id.item() == eos_id:
            break
        input_ids = torch.cat([input_ids, next_id], dim=1)

    generated_ids = input_ids[0, len(ids) :].tolist()
    return tokenizer.decode(generated_ids)


@torch.no_grad()
def generate_temperature(
    model: DecoderOnlyTransformer,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    max_new_tokens: int,
    device: str,
    temperature: float,
    eos_id: int | None = None,
) -> str:
    """Sample a continuation for ``prompt`` one token at a time, from the full
    (untruncated) softmax distribution scaled by ``1/temperature``. See module
    docstring for the role of the three required temperature settings.
    """
    model.eval()
    ids = tokenizer.encode(prompt, out_type=int)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Model has a hard max_seq_len (RoPE's rotation table, like the earlier
        # sinusoidal table, is only precomputed that far) -- truncate to the most
        # recent max_seq_len tokens so a long generation doesn't exceed it.
        context = input_ids[:, -model.max_seq_len :]
        logits = model(context)  # (1, T, vocab_size)
        probs = F.softmax(logits[0, -1, :] / temperature, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)

        if eos_id is not None and sampled.item() == eos_id:
            break
        input_ids = torch.cat([input_ids, sampled.view(1, 1)], dim=1)

    generated_ids = input_ids[0, len(ids) :].tolist()
    return tokenizer.decode(generated_ids)


def load_model_and_tokenizer(
    checkpoint_path: Path, tokenizer_path: Path, device: str | None = None
) -> tuple[DecoderOnlyTransformer, spm.SentencePieceProcessor, str]:
    """Convenience loader: checkpoint -> (model, tokenizer, device), model already
    in eval mode on the resolved device."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _, _ = load_checkpoint(checkpoint_path, device=device)
    model.to(device)
    model.eval()
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    return model, tokenizer, device
