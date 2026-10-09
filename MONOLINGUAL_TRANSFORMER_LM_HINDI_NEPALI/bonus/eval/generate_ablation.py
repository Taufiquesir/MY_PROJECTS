"""Autoregressive text generation from an ablation (no-positional-embedding)
checkpoint.

Structurally identical to ``hindi/eval/generate.py``, the only difference being
which checkpoint-loading function and model class it points at -- the actual
greedy/temperature decoding logic is copied unchanged, since generation itself
has nothing to do with which positional scheme the model uses.
"""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

from bonus.model.checkpoint_ablation import load_checkpoint
from bonus.model.transformer_ablation import DecoderOnlyTransformerNoPos


@torch.no_grad()
def generate_greedy(
    model: DecoderOnlyTransformerNoPos,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    max_new_tokens: int,
    device: str,
    eos_id: int | None = None,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt, out_type=int)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = input_ids[:, -model.max_seq_len :]
        logits = model(context)
        next_id = torch.argmax(logits[0, -1, :]).view(1, 1)

        if eos_id is not None and next_id.item() == eos_id:
            break
        input_ids = torch.cat([input_ids, next_id], dim=1)

    generated_ids = input_ids[0, len(ids) :].tolist()
    return tokenizer.decode(generated_ids)


@torch.no_grad()
def generate_temperature(
    model: DecoderOnlyTransformerNoPos,
    tokenizer: spm.SentencePieceProcessor,
    prompt: str,
    max_new_tokens: int,
    device: str,
    temperature: float,
    eos_id: int | None = None,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt, out_type=int)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = input_ids[:, -model.max_seq_len :]
        logits = model(context)
        probs = F.softmax(logits[0, -1, :] / temperature, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)

        if eos_id is not None and sampled.item() == eos_id:
            break
        input_ids = torch.cat([input_ids, sampled.view(1, 1)], dim=1)

    generated_ids = input_ids[0, len(ids) :].tolist()
    return tokenizer.decode(generated_ids)


def load_model_and_tokenizer(
    checkpoint_path: Path, tokenizer_path: Path, device: str | None = None
) -> tuple[DecoderOnlyTransformerNoPos, spm.SentencePieceProcessor, str]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _, _ = load_checkpoint(checkpoint_path, device=device)
    model.to(device)
    model.eval()
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    return model, tokenizer, device
