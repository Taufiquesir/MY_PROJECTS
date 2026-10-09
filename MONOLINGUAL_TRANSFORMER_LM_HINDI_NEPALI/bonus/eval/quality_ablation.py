"""Generation-quality (BLEU/chrF/ROUGE-L) and diversity diagnostics for the
ablation checkpoint.

Reuses everything from ``hindi/eval/quality.py`` that is not tied to a specific
model class -- the dataclasses, ``build_prompt_reference_pairs``, ``distinct_n``,
``repetition_rate``, and ``_score_condition`` are all pure text/metric logic with
no dependency on which positional scheme produced the generations, so they are
imported and reused unchanged. Only ``evaluate_quality`` itself is redefined
here, calling this ablation's own ``generate_greedy`` / ``generate_temperature`` /
``load_model_and_tokenizer`` (from ``generate_ablation.py``) instead of the real
model's.
"""

from __future__ import annotations

from pathlib import Path

from hindi.eval.quality import (  # noqa: F401 -- re-exported, unchanged
    TEMPERATURES,
    DecodingConditionResult,
    GenerationSample,
    QualityResult,
    build_prompt_reference_pairs,
    distinct_n,
    repetition_rate,
    _score_condition,
)

from bonus.eval.generate_ablation import generate_greedy, generate_temperature, load_model_and_tokenizer


def evaluate_quality(
    checkpoint_path: Path,
    tokenizer_path: Path,
    test_records: list[str],
    n_samples: int = 100,
    device: str | None = None,
) -> QualityResult:
    """Run the full generation-quality + diversity evaluation for the ablation
    checkpoint, under greedy decoding and temperature sampling at 0.5/1.0/1.5
    (four conditions total) -- identical protocol to
    ``hindi.eval.quality.evaluate_quality``, just pointed at the ablation's own
    generation primitives.
    """
    model, tokenizer, device = load_model_and_tokenizer(checkpoint_path, tokenizer_path)
    pairs = build_prompt_reference_pairs(test_records, tokenizer, n_samples)
    prompts = [p for p, _ in pairs]
    references = [r for _, r in pairs]

    by_condition: dict[str, DecodingConditionResult] = {}

    greedy_generations = [
        generate_greedy(model, tokenizer, prompt, max_new_tokens=len(tokenizer.encode(ref, out_type=int)), device=device)
        for prompt, ref in pairs
    ]
    by_condition["greedy"] = _score_condition("greedy", greedy_generations, references, prompts)

    for temp in TEMPERATURES:
        temp_generations = [
            generate_temperature(
                model, tokenizer, prompt, max_new_tokens=len(tokenizer.encode(ref, out_type=int)), device=device, temperature=temp
            )
            for prompt, ref in pairs
        ]
        by_condition[f"temperature={temp}"] = _score_condition(f"temperature={temp}", temp_generations, references, prompts)

    return QualityResult(n_samples=len(pairs), by_condition=by_condition)
