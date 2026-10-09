"""Generate training-curve plots (loss, val PPL, LR schedule) for each language
from its real training log, plus Hindi-vs-Nepali comparison plots -- run once
both languages' report/phase2/<lang>/training_log.txt exist.

Usage (from the repo root)::

    python run_training_curves.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from hindi.eval.parse_training_log import parse_training_log as parse_hindi
    from hindi.eval.plots import (
        plot_h_vs_l_bar_metrics,
        plot_h_vs_l_val_loss,
        plot_h_vs_l_val_ppl,
        plot_lr_schedule,
        plot_training_curve,
        plot_val_perplexity_curve,
    )
    from nepali.eval.parse_training_log import parse_training_log as parse_nepali

    hindi_log = REPO_ROOT / "report" / "phase2" / "hindi" / "training_log.txt"
    nepali_log = REPO_ROOT / "report" / "phase2" / "nepali" / "training_log.txt"

    hindi_curve = parse_hindi(hindi_log)
    nepali_curve = parse_nepali(nepali_log)
    print(f"hindi: {len(hindi_curve.train_steps)} train points, {len(hindi_curve.val_steps)} val points")
    print(f"nepali: {len(nepali_curve.train_steps)} train points, {len(nepali_curve.val_steps)} val points")

    hindi_out = REPO_ROOT / "report" / "phase2" / "hindi" / "plots"
    nepali_out = REPO_ROOT / "report" / "phase2" / "nepali" / "plots"

    plot_training_curve(
        hindi_curve.train_steps, hindi_curve.train_loss, hindi_curve.val_steps, hindi_curve.val_loss,
        "hindi", hindi_out / "training_loss_curve.png",
    )
    plot_val_perplexity_curve(hindi_curve.val_steps, hindi_curve.val_ppl, "hindi", hindi_out / "val_perplexity_curve.png")
    plot_lr_schedule(hindi_curve.train_steps, hindi_curve.lr, "hindi", hindi_out / "lr_schedule.png")

    plot_training_curve(
        nepali_curve.train_steps, nepali_curve.train_loss, nepali_curve.val_steps, nepali_curve.val_loss,
        "nepali", nepali_out / "training_loss_curve.png",
    )
    plot_val_perplexity_curve(nepali_curve.val_steps, nepali_curve.val_ppl, "nepali", nepali_out / "val_perplexity_curve.png")
    plot_lr_schedule(nepali_curve.train_steps, nepali_curve.lr, "nepali", nepali_out / "lr_schedule.png")

    # --- Cross-language comparison plots, into a shared comparison/ folder ---
    comparison_out = REPO_ROOT / "report" / "phase2" / "comparison"
    plot_h_vs_l_val_loss(
        hindi_curve.val_steps, hindi_curve.val_loss,
        nepali_curve.val_steps, nepali_curve.val_loss,
        comparison_out / "val_loss_hindi_vs_nepali.png",
    )
    plot_h_vs_l_val_ppl(
        hindi_curve.val_steps, hindi_curve.val_ppl,
        nepali_curve.val_steps, nepali_curve.val_ppl,
        comparison_out / "val_ppl_hindi_vs_nepali.png",
    )

    # Headline-metric bar chart, pulled from each language's metrics.json (the
    # real evaluated numbers, not the training-time validation numbers above).
    hindi_metrics = json.loads((REPO_ROOT / "report" / "phase2" / "hindi" / "metrics.json").read_text())
    nepali_metrics = json.loads((REPO_ROOT / "report" / "phase2" / "nepali" / "metrics.json").read_text())
    bar_metrics = {
        "PPL": {"Hindi": hindi_metrics["perplexity"]["ppl"], "Nepali": nepali_metrics["perplexity"]["ppl"]},
        "BPB": {"Hindi": hindi_metrics["perplexity"]["bpb"], "Nepali": nepali_metrics["perplexity"]["bpb"]},
        "chrF (greedy)": {
            "Hindi": hindi_metrics["generation_quality"]["greedy"]["chrf"],
            "Nepali": nepali_metrics["generation_quality"]["greedy"]["chrf"],
        },
        "Distinct-2 (temp=1.0)": {
            "Hindi": hindi_metrics["generation_quality"]["temperature=1.0"]["distinct_2"],
            "Nepali": nepali_metrics["generation_quality"]["temperature=1.0"]["distinct_2"],
        },
    }
    plot_h_vs_l_bar_metrics(bar_metrics, comparison_out / "headline_metrics_bar.png")

    print(f"wrote training-curve plots to {hindi_out}, {nepali_out}, and {comparison_out}")


if __name__ == "__main__":
    main()
