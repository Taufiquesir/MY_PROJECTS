"""Bonus ablation: Model H (Hindi) with every positional signal removed.

This lives entirely under ``bonus/`` and is never imported by any graded Phase 1-3
code -- the real ``hindi/model/transformer.py`` is not touched or depended on for
its attention/model classes (only its ``FeedForward`` sublayer is reused, since
that piece has nothing to do with position). Every other line here is either
identical to the real model or has one specific, called-out difference: RoPE's
rotation is never applied.

What "no positional embeddings" means here concretely: :class:`CausalSelfAttention`
below never rotates Q or K -- they go into the attention dot product exactly as the
QKV projection produced them. There was never an additive positional table to
remove in the first place, since the real Hindi model uses RoPE (rotation inside
every attention layer), not learned/sinusoidal absolute embeddings added once at
the input -- see ``hindi/model/transformer.py``'s module docstring for that
architecture's own justification. The practical effect: with no rotation and no
additive code anywhere, ``softmax(QK^T / sqrt(d_head) + M) V`` becomes a pure
function of token *content* and the causal mask's coarse before/after ordering
alone. The model has no way to tell how far apart two tokens are, or in which
order two tokens before the causal boundary occurred, beyond "one came earlier
than the other" (which the mask still enforces) -- it cannot tell *how much*
earlier. Two sentences using the same bag of words in a different order become
much harder for a single attention layer to tell apart.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hindi.model.transformer import FeedForward  # noqa: E402 -- position-independent, reused as-is


class CausalSelfAttentionNoPos(nn.Module):
    """Identical to ``hindi.model.transformer.CausalSelfAttention`` except the two
    ``apply_rope`` calls that model makes are simply never made here -- Q and K
    are used exactly as the QKV projection produced them. Every other line (QKV
    projection, causal mask construction and application, 1/sqrt(d_head) scaling,
    softmax, output projection, dropout placement) is unchanged, so any behavioral
    difference between this class and the real one is attributable only to the
    missing rotation.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # No RoPE cos/sin buffers at all -- there is no rotation table to
        # precompute, since this class never rotates Q/K. This is the entire
        # ablation: these missing buffers plus the two missing apply_rope calls
        # in forward() below, nothing else differs from the real model.
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d_model = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # <-- the ablation: no apply_rope(q, ...) / apply_rope(k, ...) here.
        # q and k carry only token content, no positional signal.

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.d_head)

        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, d_model)

        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out


class TransformerBlockNoPos(nn.Module):
    """Same pre-norm residual structure as ``hindi.model.transformer.
    TransformerBlock``, built from the real, position-independent
    :class:`FeedForward` and :class:`CausalSelfAttentionNoPos` in place of the
    real attention module.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttentionNoPos(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class DecoderOnlyTransformerNoPos(nn.Module):
    """Same overall shape as ``hindi.model.transformer.DecoderOnlyTransformer``
    (embedding -> N blocks -> final norm -> tied output head), built from
    :class:`TransformerBlockNoPos` instead of the real RoPE-using block. Kept as a
    fully separate class under ``bonus/`` (not a subclass or a flag on the real
    model) so the graded model's ``forward`` can never be affected by anything
    defined here -- reading this one file top to bottom is reading the complete
    ablation.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlockNoPos(d_model, n_heads, d_ff, max_seq_len, dropout) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        if T > self.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len={self.max_seq_len}")

        x = self.token_embedding(input_ids)
        x = self.embed_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        logits = self.lm_head(x)
        return logits

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)
