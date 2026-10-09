"""Plotting for the ablation's attention analysis.

Every plotting function in ``hindi/eval/plots.py`` only ever reads
``capture.weights`` / ``capture.tokens`` off whatever ``AttentionCapture``-shaped
object it's given -- it never imports or checks against a specific model class.
Since ``bonus.eval.attention_analysis_ablation.AttentionCapture`` has the exact
same two fields, the real plotting module works unmodified against ablation
captures, so this file just re-exports it rather than duplicating ~300 lines of
matplotlib code for a purely visual layer that has nothing to do with model
architecture.
"""

from hindi.eval.plots import (  # noqa: F401
    plot_attention_heatmap,
    plot_attention_heatmap_multihead,
    plot_distance_by_layer,
    plot_entropy_by_layer,
)
