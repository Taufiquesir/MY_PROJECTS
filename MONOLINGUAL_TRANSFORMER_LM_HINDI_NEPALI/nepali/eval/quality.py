"""Generation-quality metrics (BLEU, chrF, ROUGE-L) and diversity diagnostics
(repetition rate, Distinct-1/2) for a trained checkpoint's generated continuations.

Evaluation protocol, matching the assignment spec exactly: take real held-out
documents from the test split, split each into a prompt (first ``prompt_tokens``
tokens) and a reference continuation (the rest, up to ``max_ref_tokens``), generate
a model continuation from the same prompt under **greedy decoding and under
temperature sampling at 0.5, 1.0, and 1.5** (four decoding conditions total, see
``eval/generate.py``), and score each condition's generated text against the same
reference. This is a completion-quality measure, not a translation/summarization
measure -- BLEU/chrF/ROUGE-L are borrowed from those tasks here simply as standard
n-gram-overlap metrics for "does the model's continuation resemble what a human
writer actually wrote next," which is a reasonable proxy for a from-scratch LM at
this scale (no held-out reference set of "ideal" continuations exists to evaluate
against instead).

Metric informativeness for open-ended LM generation (why report all three, and
what each one is/isn't telling you):
- **BLEU** rewards exact n-gram overlap with the reference and is precision-
  weighted -- for open-ended continuation (many equally-plausible continuations
  exist for the same prompt, unlike translation where there's one target meaning)
  BLEU systematically under-rewards a fluent but differently-worded continuation,
  so a low BLEU here does not necessarily mean the generation is bad.
- **chrF** operates on character n-grams instead of word n-grams, which is more
  forgiving of morphological variation (a real concern for Hindi/Nepali, both
  morphologically rich) -- two inflected forms of the same word share character
  n-grams even when they don't match as whole-word tokens, so chrF is a somewhat
  more informative signal than BLEU for these languages specifically.
- **ROUGE-L** rewards the longest common subsequence rather than exact n-gram
  matches, so it's more tolerant of reordering/insertions than BLEU -- useful as a
  softer complement, but shares BLEU's "only one reference" limitation for
  open-ended generation.
None of the three is a strong absolute quality signal in this setting; they're
most informative compared *relative to each other* (Model H vs Model L, or
decoding condition vs decoding condition) rather than read as absolute scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import sentencepiece as spm
import torch
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU, CHRF

from .generate import generate_greedy, generate_temperature, load_model_and_tokenizer

TEMPERATURES = (0.5, 1.0, 1.5)


@dataclass
class GenerationSample:
    prompt: str
    reference: str
    generated: str


@dataclass
class DecodingConditionResult:
    condition: str  # "greedy" or "temperature=<t>"
    bleu: float
    chrf: float
    rouge_l_f1: float
    distinct_1: float
    distinct_2: float
    repetition_rate: float
    samples: list[GenerationSample] = field(default_factory=list)


@dataclass
class QualityResult:
    n_samples: int
    by_condition: dict[str, DecodingConditionResult]


def build_prompt_reference_pairs(
    test_records: list[str],
    tokenizer: spm.SentencePieceProcessor,
    n_samples: int,
    prompt_tokens: int = 32,
    max_ref_tokens: int = 64,
) -> list[tuple[str, str]]:
    """Build ``n_samples`` (prompt, reference) pairs from real held-out documents,
    shared across every decoding condition so all four conditions are scored
    against the exact same prompts/references. Documents shorter than
    ``prompt_tokens + 8`` tokens are skipped (too little text for a meaningful
    prompt/reference split)."""
    pairs: list[tuple[str, str]] = []
    for text in test_records:
        if len(pairs) >= n_samples:
            break
        ids = tokenizer.encode(text, out_type=int)
        if len(ids) < prompt_tokens + 8:
            continue
        prompt_ids = ids[:prompt_tokens]
        ref_ids = ids[prompt_tokens : prompt_tokens + max_ref_tokens]
        pairs.append((tokenizer.decode(prompt_ids), tokenizer.decode(ref_ids)))
    return pairs


def distinct_n(texts: list[str], n: int) -> float:
    """Distinct-n: ratio of unique n-grams to total n-grams across all ``texts``,
    a standard diversity diagnostic -- low values mean the model is repeating the
    same n-grams across (or within) its generations rather than producing varied
    text. Whitespace tokenization (not the model's own subword tokenizer), since
    this is meant to measure surface-level word/phrase repetition the way a human
    reader would notice it, not subword-token repetition.
    """
    all_ngrams: list[tuple[str, ...]] = []
    for text in texts:
        words = text.split()
        all_ngrams.extend(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def repetition_rate(texts: list[str], n: int = 4) -> float:
    """Fraction of generations that contain at least one repeated n-gram (default
    4-gram) within themselves -- a direct symptom of the degenerate short-loop
    behavior low-temperature/greedy decoding tends to produce, complementing
    Distinct-n (which measures repetition *across* the whole sample, not within a
    single generation).
    """
    if not texts:
        return 0.0
    repeated_count = 0
    for text in texts:
        words = text.split()
        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        if len(ngrams) != len(set(ngrams)):
            repeated_count += 1
    return repeated_count / len(texts)


def _score_condition(
    condition_name: str,
    generations: list[str],
    references: list[str],
    prompts: list[str],
) -> DecodingConditionResult:
    """Compute BLEU (corpus-level), chrF, ROUGE-L (sentence-level averaged), and
    diversity diagnostics for one decoding condition's generations."""
    bleu = BLEU()
    bleu_score = bleu.corpus_score(generations, [references]).score

    chrf = CHRF()
    chrf_score = chrf.corpus_score(generations, [references]).score

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l = sum(scorer.score(r, g)["rougeL"].fmeasure for g, r in zip(generations, references)) / len(generations)

    return DecodingConditionResult(
        condition=condition_name,
        bleu=bleu_score,
        chrf=chrf_score,
        rouge_l_f1=rouge_l,
        distinct_1=distinct_n(generations, 1),
        distinct_2=distinct_n(generations, 2),
        repetition_rate=repetition_rate(generations),
        samples=[GenerationSample(p, r, g) for p, r, g in zip(prompts, references, generations)],
    )


def evaluate_quality(
    checkpoint_path: Path,
    tokenizer_path: Path,
    test_records: list[str],
    n_samples: int = 100,
    device: str | None = None,
) -> QualityResult:
    """Run the full generation-quality + diversity evaluation for one checkpoint,
    under greedy decoding and temperature sampling at 0.5/1.0/1.5 (four conditions
    total), per the assignment's generation-quality protocol.

    Args:
        test_records: raw document texts from the held-out test split (not yet
            tokenized) -- typically ``[rec.text for rec in iter_records(test_dir)]``.
        n_samples: number of (prompt, reference) pairs to build and score under
            each of the four decoding conditions. Generation is the slowest part
            of this evaluation (one forward pass per output token, x4 conditions),
            so this is kept modest by default.
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
