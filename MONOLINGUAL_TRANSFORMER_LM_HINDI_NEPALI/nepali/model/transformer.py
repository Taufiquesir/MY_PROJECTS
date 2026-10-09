"""Decoder-only (GPT-style) Transformer language model, built from primitive PyTorch
layers only -- no ``nn.Transformer*``, no HuggingFace model classes, no pre-built
attention block. Every tensor shape below is explicit and commented because the
assignment requires being able to explain each operation in the forward pass.

Positional scheme: **RoPE** (Rotary Position Embedding, Su et al. 2021). Rather than
adding a positional vector to the token embedding before the first block (as
sinusoidal/learned-absolute schemes do), RoPE rotates each head's query and key
vectors by an angle proportional to their sequence position, *inside* every
attention computation. Chosen over sinusoidal/learned-absolute for two concrete
reasons: (1) it makes attention scores a function of *relative* position by
construction -- rotating both Q at position i and K at position j by their own
absolute angle leaves the Q-K dot product depending only on (i - j), not on i and j
separately, which is a strictly more direct relative-position signal than hoping a
downstream linear layer recovers relative structure from an absolute additive code;
(2) it adds no extra parameters (same as sinusoidal) and, unlike an additive
embedding table, never mixes positional and content information within a single
vector before attention even runs -- the rotation is applied freshly to Q/K in every
layer rather than being "spent" once at the input. The concrete cost, matching this
project's earlier sinusoidal implementation: rotation still uses a fixed frequency
table computed out to ``max_seq_len``, so this is the model's effective hard context
cap the same as any other scheme here (RoPE variants exist that extrapolate beyond
their training length, but this implementation does not attempt that).

Norm placement: **pre-norm** (LayerNorm applied to the sublayer's input, before
attention/FFN, with the residual add happening around the *un-normalized* input).
Chosen over post-norm because post-norm's residual stream is repeatedly renormalized
after every add, which empirically makes gradients harder to keep well-scaled as depth
grows without a carefully tuned learning-rate warmup -- pre-norm instead keeps an
un-normalized residual "highway" all the way from the embeddings to the final norm, so
gradients can flow through addition (identity) rather than through a normalization op
at every single block. At 13 layers this is a real, not just textbook, stability
margin, and removes one training-stability variable to debug on a tight compute
budget.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_frequencies(max_seq_len: int, d_head: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the cos/sin rotation tables RoPE needs, once, up to ``max_seq_len``.

    RoPE treats each head's ``d_head``-dimensional vector as ``d_head / 2`` 2D
    subvectors and rotates subvector ``i`` by angle ``pos * theta_i``, where
    ``theta_i = base^(-2i/d_head)`` -- the same geometric frequency schedule as
    sinusoidal position encoding, just applied as a rotation at attention time
    instead of as an additive code at the input.

    Returns:
        ``(cos, sin)``, each of shape ``(max_seq_len, d_head // 2)``, dtype
        float32. Not ``nn.Parameter``s -- registered as buffers by the caller,
        fixed and must move with ``.to(device)`` but never receive a gradient.
    """
    theta = base ** (-torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)  # (d_head/2,)
    position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)  # (T, 1)
    angles = position * theta.unsqueeze(0)  # (T, d_head/2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x``'s last dimension in place-equivalent fashion, per RoPE.

    Args:
        x: ``(B, h, T, d_head)`` -- query or key vectors, one head at a time
            (broadcasts over ``B`` and ``h``).
        cos, sin: ``(T, d_head // 2)``, already sliced to this call's sequence
            length by the caller.

    Returns:
        ``(B, h, T, d_head)``, same shape as ``x``.

    Splits the last dimension into even/odd-indexed halves ``x1, x2`` (each
    ``d_head // 2`` wide) and applies the standard 2D rotation
    ``(x1', x2') = (x1*cos - x2*sin, x1*sin + x2*cos)`` to every (x1[i], x2[i])
    pair independently -- this is exactly a block-diagonal rotation matrix with
    ``d_head / 2`` independent 2x2 rotation blocks, each spinning at its own
    frequency ``theta_i``.
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]  # each (B, h, T, d_head/2)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_head/2), broadcasts over B, h
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated_even = x1 * cos - x2 * sin
    rotated_odd = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rotated_even
    out[..., 1::2] = rotated_odd
    return out


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, implemented from first principles.

    Given input ``X`` of shape ``(B, T, d_model)``:
      1. Linearly project to queries/keys/values: ``Q = X W_Q``, ``K = X W_K``,
         ``V = X W_V``, each of shape ``(B, T, d_model)``.
      2. Reshape each of ``Q, K, V`` from ``(B, T, d_model)`` into
         ``(B, n_heads, T, d_head)`` so every head operates in its own
         ``d_head = d_model / n_heads``-dimensional subspace independently.
      3. Rotate ``Q`` and ``K`` (not ``V``) by RoPE, per-position -- this is where
         positional information enters the model; see the module docstring.
      4. Compute scaled dot-product attention per head:
         ``softmax(Q K^T / sqrt(d_head) + M) V``, where ``M`` is the causal mask.
      5. Reshape the per-head outputs back from ``(B, n_heads, T, d_head)`` to
         ``(B, T, d_model)`` and apply the output projection ``W_O``.
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        if self.d_head % 2 != 0:
            raise ValueError(f"d_head={self.d_head} must be even for RoPE's 2D-subvector rotation")

        # Combined QKV projection (one matmul instead of three) -- mathematically
        # identical to three separate nn.Linear(d_model, d_model) layers; the single
        # matrix is just split into three equal chunks along the output dimension
        # right after the projection.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # RoPE cos/sin tables, precomputed once up to max_seq_len and sliced per
        # forward call rather than rebuilt every time. See rope_frequencies'
        # docstring for the angle schedule. Registered as buffers (not Parameters):
        # fixed, no gradient, but must move with .to(device).
        cos, sin = rope_frequencies(max_seq_len, self.d_head)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Causal mask, precomputed once up to max_seq_len and sliced per forward call
        # rather than rebuilt every time. Registered as a buffer (not a Parameter):
        # it has no gradient and must move with .to(device), but is never trained.
        # torch.triu(..., diagonal=1) is upper-triangular *excluding* the main
        # diagonal, i.e. True exactly where key position > query position (the
        # future) -- those entries get -inf before softmax so they contribute
        # exactly zero probability mass to the output.
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, T, d_model)``.

        Returns:
            ``(B, T, d_model)`` -- same shape as the input, since attention is a
            sequence-to-sequence mixing operation that does not change dimensionality.
        """
        B, T, d_model = x.shape

        qkv = self.qkv_proj(x)  # (B, T, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, d_model)

        # (B, T, d_model) -> (B, T, n_heads, d_head) -> (B, n_heads, T, d_head).
        # The transpose puts n_heads before T so that the matmuls below broadcast
        # over (B, n_heads) as independent batch dimensions, each head attending
        # only within its own d_head-dimensional subspace.
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, T, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, T, d_head)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, T, d_head)

        # Rotate Q and K by their sequence position (RoPE) -- V is left untouched
        # since it carries content, not something attention needs to compare by
        # position. Sliced to the actual sequence length T (tables are precomputed
        # for max_seq_len).
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]  # each (T, d_head/2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # QK^T: (B, h, T, d_head) @ (B, h, d_head, T) -> (B, h, T, T) -- one
        # attention-score matrix per head, entry [i, j] is how much query position i
        # attends to key position j before masking/softmax. Because both q and k
        # were rotated by RoPE, this dot product depends only on (i - j), not on i
        # and j independently -- the relative-position property described in the
        # module docstring.
        scores = q @ k.transpose(-2, -1)  # (B, h, T, T)

        # Scale by 1/sqrt(d_head): without this, the dot product's variance grows
        # with d_head (each of d_head independent terms contributes ~unit variance,
        # so the sum's variance grows linearly with d_head), which pushes softmax
        # into a regime with extremely small gradients (one score dominates, the
        # rest saturate near zero) once d_head is not tiny. Dividing by sqrt(d_head)
        # keeps the scores' variance ~1 regardless of d_head, so softmax stays in a
        # well-conditioned regime.
        scores = scores / math.sqrt(self.d_head)

        # Causal mask: additive -inf on future positions (key index > query index)
        # so softmax assigns them exactly zero probability. Sliced to the actual
        # sequence length T (the buffer is precomputed for max_seq_len).
        mask = self.causal_mask[:T, :T]  # (T, T), True where masked
        scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)  # (B, h, T, T), rows sum to 1
        attn = self.attn_dropout(attn)

        # (B, h, T, T) @ (B, h, T, d_head) -> (B, h, T, d_head): each query position's
        # output is a probability-weighted sum over value vectors at all (unmasked)
        # key positions.
        out = attn @ v  # (B, h, T, d_head)

        # Undo the earlier reshape: (B, h, T, d_head) -> (B, T, h, d_head) -> (B, T,
        # d_model). .contiguous() is required because .transpose() only changes
        # strides, not memory layout, and .view() needs contiguous memory.
        out = out.transpose(1, 2).contiguous().view(B, T, d_model)

        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network: two linear layers with a GELU nonlinearity
    in between, expanding to a wider inner dimension (``d_ff > d_model``) and back.

    "Position-wise" means this is applied independently and identically to every
    sequence position -- there is no mixing across the T dimension here (that only
    happens in attention), so this sublayer purely transforms each position's own
    feature vector.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, T, d_model)`` -> ``(B, T, d_model)``."""
        x = self.fc1(x)  # (B, T, d_ff)
        x = F.gelu(x)
        x = self.fc2(x)  # (B, T, d_model)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """One decoder block: pre-norm causal self-attention, then pre-norm feed-forward,
    each wrapped in its own residual connection.

    Pre-norm means ``x = x + Sublayer(LayerNorm(x))`` rather than
    ``x = LayerNorm(x + Sublayer(x))`` -- the residual stream (the ``x`` that gets
    added back) is never itself passed through a LayerNorm, only the *input* to each
    sublayer is normalized. See the module docstring for why this is the stabler
    choice at this depth.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, T, d_model)`` -> ``(B, T, d_model)``, shape preserved throughout."""
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    """GPT-style decoder-only Transformer LM.

    Forward pass stages (see the assignment spec for the exact required structure):
      1. Token embedding, then embedding dropout. Unlike a sinusoidal/learned-
         absolute scheme, there is no positional table added here -- RoPE injects
         position inside every attention layer instead (see
         :class:`CausalSelfAttention`).
      2. ``n_layers`` stacked :class:`TransformerBlock`s (pre-norm causal
         self-attention + pre-norm feed-forward, each with a residual connection).
      3. Final LayerNorm.
      4. Linear output head projecting back to vocabulary size, producing next-token
         logits. Weight-tied with the input token embedding (see ``__init__``).
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
            [TransformerBlock(d_model, n_heads, d_ff, max_seq_len, dropout) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)

        # Output head: nn.Linear(d_model, vocab_size), but its weight matrix is
        # *tied* to token_embedding.weight (same underlying tensor, shared during
        # both forward and backward) rather than a separate learned matrix. Both
        # matrices are the same shape (vocab_size, d_model) since nn.Linear stores
        # weight as (out_features, in_features) = (vocab_size, d_model), identical
        # to nn.Embedding's (num_embeddings, embedding_dim). Saves vocab_size *
        # d_model parameters (here, 9000 * 384 = 3.456M, about 14% of the model) and
        # is a well-established trick (Press & Wolf, 2017) motivated by the
        # intuition that a token's embedding and its "how likely is this the next
        # token" direction in the output space should be related.
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Standard small-init scheme: N(0, 0.02) for linear/embedding weights, zero
        bias, unit-scale/zero-bias LayerNorm -- keeps activations well-scaled at the
        start of training rather than relying on PyTorch's default (Kaiming-uniform
        for Linear), which was tuned for much deeper conv/ReLU networks, not this.
        """
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
        """
        Args:
            input_ids: ``(B, T)`` int64 token ids, ``T <= max_seq_len``.

        Returns:
            logits: ``(B, T, vocab_size)`` -- unnormalized next-token scores at every
            position. Position ``t``'s logits predict the token at position ``t+1``
            (the causal LM objective consumes this as
            ``cross_entropy(logits[:, :-1], input_ids[:, 1:])``, not computed here).
        """
        B, T = input_ids.shape
        if T > self.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len={self.max_seq_len}")

        x = self.token_embedding(input_ids)  # (B, T, d_model)
        x = self.embed_dropout(x)

        for block in self.blocks:
            x = block(x)  # (B, T, d_model), shape unchanged through every block; each
            # block's attention applies RoPE internally using T inferred from x

        x = self.final_ln(x)  # (B, T, d_model)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Total parameter count. Weight tying means the shared embedding/head
        matrix is counted once, not twice (``model.parameters()`` already
        deduplicates by tensor identity, not by module, so tied weights are not
        double-counted here)."""
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)
