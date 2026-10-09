"""Attention analysis: heatmaps, per-head entropy, and mean attention distance.

Extracts attention weights via a `forward hook on each block's `CausalSelfAttention`
module, re-deriving the same softmax(QK^T/sqrt(d_head) + mask) computation the
model's own forward pass does internally, rather than editing
``model/transformer.py`` to return attention weights from its forward pass. Kept
fully separate on purpose: the training code stays exactly as validated (and as
currently mid-run on Kaggle) with zero risk of a change here affecting it, and
analysis/visualization logic living in its own module matches the project's
established convention (see ``collection/common/plots.py`` from Phase 1) of never
mixing computation and visualization in one function.

Under RoPE, "attention" here is still the standard post-softmax (B, h, T, T)
probability matrix -- RoPE only changes *how* Q and K are rotated before the dot
product, not the shape or interpretation of the resulting attention weights, so
every analysis below (heatmaps, entropy, mean attention distance) applies exactly
as it would for any other positional scheme.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..model.transformer import DecoderOnlyTransformer, apply_rope


@dataclass
class AttentionCapture:
    """Attention weights for every layer/head on one forward pass.

    ``weights``: list of length ``n_layers``, each entry ``(n_heads, T, T)`` --
    kept per-layer rather than stacked into one tensor since different models could
    in principle have different ``n_heads`` per layer (not the case here, but keeps
    this reusable), and because per-layer access is how every function below
    consumes it anyway.
    """

    weights: list[torch.Tensor]
    tokens: list[str]  # the actual token strings attended over, for axis labels


@torch.no_grad()
def capture_attention(
    model: DecoderOnlyTransformer,
    input_ids: torch.Tensor,
    tokenizer,
    device: str,
) -> AttentionCapture:
    """Run one forward pass, recomputing each layer's attention weights alongside it.

    Args:
        input_ids: ``(1, T)`` -- single sequence (batch size 1), since heatmaps and
            per-position distance metrics are inspected one example at a time.
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

        cos, sin = attn_module.rope_cos[:T], attn_module.rope_sin[:T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(attn_module.d_head)
        mask = attn_module.causal_mask[:T, :T]
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (1, h, T, T)
        all_weights.append(weights.squeeze(0).cpu())  # (h, T, T)

        # Advance x through the real block so later layers see the real residual
        # stream, not a recomputation drifted from what the actual forward pass
        # would produce.
        x = x + block.attn(ln_x)
        x = x + block.ffn(block.ln2(x))

    tokens = [tokenizer.id_to_piece(int(i)) for i in input_ids[0].tolist()]
    return AttentionCapture(weights=all_weights, tokens=tokens)


def attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) of each query position's attention distribution.

    Args:
        weights: ``(h, T, T)`` or ``(T, T)`` -- one layer's (all heads', or one
            head's) post-softmax attention weights.

    Returns:
        Same leading shape minus the last dim: ``(h, T)`` or ``(T,)`` -- one entropy
        value per (head, query position). Low entropy = the position attends
        sharply to a few keys; high entropy (up to ``log(T)`` for a fully causal-
        unmasked-equivalent uniform distribution) = attention is spread broadly.
        Masked (zero-probability, from -inf before softmax) entries contribute 0 to
        the sum by convention (0 * log(0) := 0), which is what makes summing over
        the full T-length last axis correct even though early positions can only
        attend to a handful of unmasked keys.
        """
    eps = 1e-12
    return -(weights * (weights + eps).log()).sum(dim=-1)


def mean_attention_distance(weights: torch.Tensor) -> torch.Tensor:
    """Average |query_position - key_position|, weighted by attention probability.

    Args:
        weights: ``(h, T, T)`` -- one layer's attention weights.

    Returns:
        ``(h,)`` -- one value per head: the average absolute distance (in token
        positions) that head's attention mass reaches back to, averaged over all
        query positions. A head that mostly attends to the immediately preceding
        token has a small mean distance; a head that reaches far back into context
        has a large one -- this is a standard diagnostic for whether different
        heads specialize into "local" vs "long-range" attention patterns.
    """
    h, T, _ = weights.shape
    positions = torch.arange(T, dtype=torch.float32)
    distance = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()  # (T, T)
    weighted = weights * distance.unsqueeze(0)  # (h, T, T)
    # Average over query positions too (dim=1), after summing weighted distance
    # over keys (dim=-1) -- each query position's attention already sums to ~1
    # (causal masking) so this is a proper weighted average per head.
    return weighted.sum(dim=-1).mean(dim=-1)


def summarize_attention(capture: AttentionCapture) -> dict:
    """Numeric summary (entropy, mean distance) for every layer/head -- the
    plotting functions in ``eval/plots.py`` consume this, kept as plain
    Python/numpy so it's trivial to serialize to the report's JSON/tables without
    depending on torch there.
    """
    n_layers = len(capture.weights)
    entropy_by_layer = []
    distance_by_layer = []
    for layer_weights in capture.weights:
        entropy_by_layer.append(attention_entropy(layer_weights).mean(dim=-1).numpy())  # (h,)
        distance_by_layer.append(mean_attention_distance(layer_weights).numpy())  # (h,)
    return {
        "n_layers": n_layers,
        "entropy_by_layer": [e.tolist() for e in entropy_by_layer],  # [n_layers][n_heads]
        "distance_by_layer": [d.tolist() for d in distance_by_layer],
    }
