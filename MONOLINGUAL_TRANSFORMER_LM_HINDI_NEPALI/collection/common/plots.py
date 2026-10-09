"""Matplotlib figures for the Phase 1 report.

Every function here only draws a figure from data passed in -- none of them compute
statistics themselves, per the assignment's requirement that visualization and
computation logic live in separate functions. Every axis is labelled and every figure
titled; plots with more than one series carry a legend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_source_contribution(domain_bytes: dict[str, int], out_path: Path, lang: str) -> None:
    """Bar chart of manual-collection bytes contributed by each crawled domain."""
    names = list(domain_bytes.keys())
    values = [v / 1e6 for v in domain_bytes.values()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, values, color="#4C72B0")
    ax.set_title(f"{lang.capitalize()}: manual-collection contribution by source")
    ax.set_xlabel("Source domain")
    ax.set_ylabel("Cleaned text (MB)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_manual_vs_downloaded(manual_bytes: int, downloaded_bytes: int, out_path: Path, lang: str) -> None:
    """Pie chart of manual vs. downloaded token/byte share."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [manual_bytes, downloaded_bytes],
        labels=["Manual", "Downloaded"],
        autopct="%1.1f%%",
        colors=["#DD8452", "#4C72B0"],
    )
    ax.set_title(f"{lang.capitalize()}: manual vs. downloaded (by bytes)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_filter_funnel(stage_counts: dict[str, int], out_path: Path, lang: str) -> None:
    """Horizontal bar chart showing document counts surviving each cleaning stage."""
    stages = list(stage_counts.keys())
    counts = list(stage_counts.values())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(stages, counts, color="#55A868")
    ax.set_title(f"{lang.capitalize()}: cleaning filter funnel")
    ax.set_xlabel("Documents remaining")
    ax.set_ylabel("Pipeline stage")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fertility_sweep(sweep_results: dict[int, dict], out_path: Path, lang: str) -> None:
    """Line plot of tokenizer fertility (chars/token) across candidate vocab sizes."""
    sizes = sorted(sweep_results.keys())
    fertility = [sweep_results[s]["fertility_chars_per_token"] for s in sizes]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, fertility, marker="o", color="#C44E52", label="chars/token")
    ax.set_title(f"{lang.capitalize()}: tokenizer fertility vs. vocabulary size")
    ax.set_xlabel("Vocabulary size")
    ax.set_ylabel("Characters per token (fertility)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_length_histogram(lengths: list[int], out_path: Path, lang: str) -> None:
    """Histogram of document lengths (characters) in the cleaned corpus."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(lengths, bins=50, color="#8172B2")
    ax.set_title(f"{lang.capitalize()}: document length distribution")
    ax.set_xlabel("Document length (characters)")
    ax.set_ylabel("Number of documents")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_zipf(token_frequencies: list[int], out_path: Path, lang: str) -> None:
    """Log-log rank-frequency plot of token frequencies (Zipf's law check)."""
    ranks = list(range(1, len(token_frequencies) + 1))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ranks, token_frequencies, color="#4C72B0", label="observed")
    ax.set_title(f"{lang.capitalize()}: token frequency rank (Zipf plot)")
    ax.set_xlabel("Rank (log scale)")
    ax.set_ylabel("Frequency (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
