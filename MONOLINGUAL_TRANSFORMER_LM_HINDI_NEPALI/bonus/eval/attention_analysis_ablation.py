"""Attention analysis for the ablation (no-positional-embedding) checkpoint:
heatmaps, per-head entropy, and mean attention distance.

Identical to ``hindi/eval/attention_analysis.py`` except ``capture_attention``
below never calls ``apply_rope`` (there is no such function to call for this
model -- see ``bonus/model/transformer_ablation.py``), matching exactly how the
ablation model's own real forward pass differs from the standard model's. Every
other line -- the QKV projection, the mask, the scaling, the softmax, advancing
``x`` through the real block afterward -- is unchanged, and the entropy / mean
attention distance formulas below are the identical, model-agnostic math from the
real analysis module (they operate purely on the post-softmax (h, T, T) weights,
with no dependence on how those weights were produced).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bonus.model.transformer_ablation import DecoderOnlyTransformerNoPos


@dataclass
class AttentionCapture:
    """Attention weights for every layer/head on one forward pass. Same shape
    contract as ``hindi.eval.attention_analysis.AttentionCapture`` -- this is a
    plain data-holding class, so ``hindi/eval/plots.py``'s plotting functions
    (which only ever read ``.weights``/``.tokens``) work unmodified against
    captures produced here."""

    weights: list[torch.Tensor]
    tokens: list[str]


@torch.no_grad()
def capture_attention(
    model: DecoderOnlyTransformerNoPos,
    input_ids: torch.Tensor,
    tokenizer,
    device: str,
) -> AttentionCapture:
    """Run one forward pass, recomputing each layer's attention weights alongside it.

    Args:
        input_ids: ``(1, T)`` -- single sequence (batch size 1).
    """
    model.eval()
    input_ids = input_ids.to(device)
    B, T = input_ids.shape
    assert B == 1, "capture_attention expects a single sequence (batch size 1)"

    all_weights: list[torch.Tensor] = []

    x = model.token_embedding(input_ids)
    x = model.embed_dropout(x)

    for block in model.blocks:
        ln_x = block.ln1(x)
        attn_module = block.attn

        qkv = attn_module.qkv_proj(ln_x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, attn_module.n_heads, attn_module.d_head).transpose(1, 2)
        k = k.view(B, T, attn_module.n_heads, attn_module.d_head).transpose(1, 2)
        v = v.view(B, T, attn_module.n_heads, attn_module.d_head).transpose(1, 2)

        # <-- the ablation: no apply_rope(q, ...) / apply_rope(k, ...) here,
        # matching CausalSelfAttentionNoPos.forward exactly.

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(attn_module.d_head)
        mask = attn_module.causal_mask[:T, :T]
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        all_weights.append(weights.squeeze(0).cpu())

        x = x + block.attn(ln_x)
        x = x + block.ffn(block.ln2(x))

    tokens = [tokenizer.id_to_piece(int(i)) for i in input_ids[0].tolist()]
    return AttentionCapture(weights=all_weights, tokens=tokens)


def attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) of each query position's attention distribution.
    Identical formula to the real model's analysis -- see
    ``hindi.eval.attention_analysis.attention_entropy`` for the full explanation.
    """
    eps = 1e-12
    return -(weights * (weights + eps).log()).sum(dim=-1)


def mean_attention_distance(weights: torch.Tensor) -> torch.Tensor:
    """Average |query_position - key_position|, weighted by attention probability.
    Identical formula to the real model's analysis -- see
    ``hindi.eval.attention_analysis.mean_attention_distance`` for the full
    explanation.
    """
    h, T, _ = weights.shape
    positions = torch.arange(T, dtype=torch.float32)
    distance = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()
    weighted = weights * distance.unsqueeze(0)
    return weighted.sum(dim=-1).mean(dim=-1)


def summarize_attention(capture: AttentionCapture) -> dict:
    """Numeric summary (entropy, mean distance) for every layer/head."""
    n_layers = len(capture.weights)
    entropy_by_layer = []
    distance_by_layer = []
    for layer_weights in capture.weights:
        entropy_by_layer.append(attention_entropy(layer_weights).mean(dim=-1).numpy())
        distance_by_layer.append(mean_attention_distance(layer_weights).numpy())
    return {
        "n_layers": n_layers,
        "entropy_by_layer": [e.tolist() for e in entropy_by_layer],
        "distance_by_layer": [d.tolist() for d in distance_by_layer],
    }
