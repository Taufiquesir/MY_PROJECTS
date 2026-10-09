"""Plotting for Phase 2 attention analysis. Pure visualization -- all computation
lives in ``eval/attention_analysis.py`` (see that module's docstring for why the
split matters). Every plot has a title, axis labels, and a legend where applicable,
per the assignment's explicit plotting requirement.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .attention_analysis import AttentionCapture


def plot_attention_heatmap(
    capture: AttentionCapture,
    layer_idx: int,
    head_idx: int,
    lang: str,
    out_path: Path,
) -> None:
    """One layer/head's (T, T) attention matrix as a heatmap, tokens labeled on
    both axes. Query positions on the y-axis (rows), key positions on the x-axis
    (columns) -- entry (i, j) is how much query token i attends to key token j.
    """
    weights = capture.weights[layer_idx][head_idx].numpy()  # (T, T)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(weights, cmap="viridis", aspect="auto")
    ax.set_title(f"{lang.capitalize()} -- Attention Heatmap (Layer {layer_idx}, Head {head_idx})")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    n = len(capture.tokens)
    if n <= 30:  # only label individual tokens when the sequence is short enough to read
        ax.set_xticks(range(n))
        ax.set_xticklabels(capture.tokens, rotation=90, fontsize=6)
        ax.set_yticks(range(n))
        ax.set_yticklabels(capture.tokens, fontsize=6)
    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_attention_heatmap_multihead(
    capture: AttentionCapture,
    layer_idx: int,
    head_indices: list[int],
    lang: str,
    out_path: Path,
) -> None:
    """One layer's attention heatmaps across multiple heads, side by side in a
    grid -- the assignment explicitly asks for heatmaps across "multiple heads",
    not just one representative head per layer, since different heads within the
    same layer can specialize into very different attention patterns (see
    eval/attention_analysis.py's mean_attention_distance for the numeric version
    of this same comparison).
    """
    n_heads = len(head_indices)
    n_cols = min(n_heads, 3)
    n_rows = -(-n_heads // n_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False)
    n = len(capture.tokens)

    for idx, head_idx in enumerate(head_indices):
        ax = axes[idx // n_cols][idx % n_cols]
        weights = capture.weights[layer_idx][head_idx].numpy()
        im = ax.imshow(weights, cmap="viridis", aspect="auto")
        ax.set_title(f"Head {head_idx}")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        if n <= 20:
            ax.set_xticks(range(n))
            ax.set_xticklabels(capture.tokens, rotation=90, fontsize=5)
            ax.set_yticks(range(n))
            ax.set_yticklabels(capture.tokens, fontsize=5)
        fig.colorbar(im, ax=ax, label="Attention weight", fraction=0.046)

    # Hide any unused grid cells (n_heads not a multiple of n_cols).
    for idx in range(n_heads, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(f"{lang.capitalize()} -- Attention Heatmaps, Layer {layer_idx} (all heads)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_entropy_by_layer(entropy_by_layer: list[list[float]], lang: str, out_path: Path) -> None:
    """Mean attention entropy per layer, one line per head -- shows whether
    entropy (how spread-out attention is) systematically rises or falls with
    depth, and whether heads within a layer specialize differently.
    """
    entropy = np.array(entropy_by_layer)  # (n_layers, n_heads)
    n_layers, n_heads = entropy.shape
    fig, ax = plt.subplots(figsize=(8, 5))
    for h in range(n_heads):
        ax.plot(range(n_layers), entropy[:, h], marker="o", label=f"Head {h}")
    ax.set_title(f"{lang.capitalize()} -- Attention Entropy by Layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean entropy (nats)")
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_distance_by_layer(distance_by_layer: list[list[float]], lang: str, out_path: Path) -> None:
    """Mean attention distance per layer, one line per head -- shows whether heads
    specialize into local (small distance) vs long-range (large distance)
    attention, and whether that specialization shifts with depth.
    """
    distance = np.array(distance_by_layer)  # (n_layers, n_heads)
    n_layers, n_heads = distance.shape
    fig, ax = plt.subplots(figsize=(8, 5))
    for h in range(n_heads):
        ax.plot(range(n_layers), distance[:, h], marker="o", label=f"Head {h}")
    ax.set_title(f"{lang.capitalize()} -- Mean Attention Distance by Layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean |query pos - key pos| (tokens)")
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _moving_average(values: list[float], window: int) -> np.ndarray:
    """Simple trailing moving average, used only to draw a smoothed overlay on top
    of the raw (noisy, logged every 50 steps) train-loss series -- the raw series
    is still plotted underneath at low opacity, so smoothing never hides the
    actual logged values, just makes the overall trend easier to read by eye.
    """
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_training_curve(
    train_steps: list[int],
    train_loss: list[float],
    val_steps: list[int],
    val_loss: list[float],
    lang: str,
    out_path: Path,
    smooth_window: int = 20,
) -> None:
    """Train/val loss vs. training step, parsed from the real training log via
    eval/parse_training_log.py -- the primary evidence that training actually
    converged, required in the report. train_steps/val_steps are each series'
    own actual logged step numbers (not assumed to align or be evenly spaced),
    since val_loss is logged far less often (every eval_every steps) than
    train_loss (every log_every steps).
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(train_steps, train_loss, alpha=0.25, color="tab:blue", label="Train loss (raw, every 50 steps)")
    if len(train_loss) >= smooth_window:
        smoothed = _moving_average(train_loss, smooth_window)
        smoothed_steps = train_steps[smooth_window - 1 :]
        ax.plot(smoothed_steps, smoothed, color="tab:blue", linewidth=2, label=f"Train loss ({smooth_window}-step moving avg)")
    if val_loss:
        ax.plot(val_steps, val_loss, marker="o", markersize=3, color="tab:orange", label="Val loss")
    ax.set_title(f"{lang.capitalize()} -- Training Loss Curve (RoPE, {max(train_steps):,} steps)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss (nats/token)")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_val_perplexity_curve(val_steps: list[int], val_ppl: list[float], lang: str, out_path: Path) -> None:
    """Validation perplexity vs. training step -- the same convergence evidence
    as the loss curve, in the units (PPL) most directly comparable to the final
    reported PPL/BPB numbers.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(val_steps, val_ppl, marker="o", markersize=4, color="tab:green")
    ax.set_title(f"{lang.capitalize()} -- Validation Perplexity During Training")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation perplexity")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lr_schedule(train_steps: list[int], lr: list[float], lang: str, out_path: Path) -> None:
    """Learning rate vs. training step -- shows the linear-warmup-then-cosine-
    decay schedule actually applied, direct evidence the LR scheduling
    requirement (assignment 2.2) was implemented as described.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(train_steps, lr, color="tab:red")
    ax.set_title(f"{lang.capitalize()} -- Learning Rate Schedule (warmup + cosine decay)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Learning rate")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_h_vs_l_val_loss(
    h_steps: list[int], h_val_loss: list[float],
    l_steps: list[int], l_val_loss: list[float],
    out_path: Path,
) -> None:
    """Model H vs Model L validation loss, same axes, direct visual comparison of
    the resource-tier gap the report is required to discuss (assignment 2.3:
    "Discuss the gap between Model H and Model L"). Not tied to one specific
    language pair by design -- callers pass in whichever two languages they used.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(h_steps, h_val_loss, marker="o", markersize=3, label="Model H (Hindi)", color="tab:blue")
    ax.plot(l_steps, l_val_loss, marker="o", markersize=3, label="Model L (Nepali)", color="tab:orange")
    ax.set_title("Model H vs Model L -- Validation Loss During Training")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation cross-entropy loss (nats/token)")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_h_vs_l_val_ppl(
    h_steps: list[int], h_val_ppl: list[float],
    l_steps: list[int], l_val_ppl: list[float],
    out_path: Path,
) -> None:
    """Model H vs Model L validation perplexity, same axes -- companion to
    plot_h_vs_l_val_loss in the units the final reported metrics use.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(h_steps, h_val_ppl, marker="o", markersize=3, label="Model H (Hindi)", color="tab:blue")
    ax.plot(l_steps, l_val_ppl, marker="o", markersize=3, label="Model L (Nepali)", color="tab:orange")
    ax.set_title("Model H vs Model L -- Validation Perplexity During Training")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation perplexity")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_h_vs_l_bar_metrics(metrics: dict[str, dict[str, float]], out_path: Path) -> None:
    """Grouped bar chart comparing Model H vs Model L on a handful of headline
    scalar metrics (e.g. final PPL, BPB) -- a compact, at-a-glance summary
    alongside the step-by-step curves above.

    Bars are normalized per-metric to Hindi=1.0 (each metric's own pair of bars
    is divided by that metric's Hindi value) rather than plotted in raw units on
    one shared axis -- PPL (~16-24) and BPB (~0.4) differ by almost two orders of
    magnitude, so a shared raw-value axis would make BPB's bars invisible next to
    PPL's. Exact values are still shown as text labels above each bar so nothing
    is lost to the normalization.

    Args:
        metrics: ``{"PPL": {"Hindi": 15.77, "Nepali": 23.78}, "BPB": {...}, ...}``
    """
    labels = list(metrics.keys())
    hindi_raw = [metrics[k].get("Hindi", 0.0) for k in labels]
    nepali_raw = [metrics[k].get("Nepali", 0.0) for k in labels]
    hindi_norm = [1.0 for _ in labels]
    nepali_norm = [(n / h if h != 0 else 0.0) for n, h in zip(nepali_raw, hindi_raw)]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.8), 5.5))
    bars_h = ax.bar(x - width / 2, hindi_norm, width, label="Model H (Hindi)", color="tab:blue")
    bars_l = ax.bar(x + width / 2, nepali_norm, width, label="Model L (Nepali)", color="tab:orange")
    for bar, val in zip(bars_h, hindi_raw):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3g}", ha="center", fontsize=8)
    for bar, val in zip(bars_l, nepali_raw):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3g}", ha="center", fontsize=8)
    ax.set_title("Model H vs Model L -- Headline Metric Comparison (normalized to Hindi = 1.0)")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value relative to Hindi (raw value labeled above each bar)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
