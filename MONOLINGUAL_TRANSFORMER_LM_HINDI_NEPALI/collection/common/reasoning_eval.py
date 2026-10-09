"""Pretrained-vs-finetuned exact-match accuracy on the synthetic reasoning test set.

The assignment asks specifically for "pretrained vs. finetuned accuracy (or exact
match) on the synthetic test set" -- not just the finetuned model's accuracy alone.
Running the *same* checkpoint-loading and generation code against both the Phase 2
pretrained checkpoint and the Phase 3 finetuned checkpoint (with the only difference
being which checkpoint is passed in) is what makes that comparison meaningful: any
accuracy gap is attributable to finetuning, not to a difference in how the two runs
were evaluated.

Exact match, not a softer metric (e.g. token overlap): every example's ground truth
is a single entity name (see ``reasoning_gen.py``), and the model's generated
continuation either names that exact entity or it doesn't -- there is no partial
credit that makes sense for "which of these three is tallest."

Accuracy is broken down per question ``kind`` (pairwise / ranking / transitive) in
addition to the overall number, since the assignment calls out multi-hop transitive
chaining as the harder skill worth reporting on specifically.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import sentencepiece as spm
import torch

from .reasoning_data import load_examples


@dataclass
class ReasoningEvalResult:
    n_examples: int
    n_correct: int
    accuracy: float
    accuracy_by_kind: dict[str, float]
    n_by_kind: dict[str, int]
    examples: list[dict]  # a sample of (prompt, gold, predicted, correct) for the report


def _normalize(text: str) -> str:
    """Strip whitespace/punctuation noise so a generated answer that's otherwise an
    exact match (e.g. trailing sentence-final punctuation the model continues to
    generate) is not marked wrong for a formatting artifact unrelated to whether the
    comparison itself was answered correctly.
    """
    return text.strip().strip("।.,!?​").strip()


@torch.no_grad()
def evaluate_reasoning_accuracy(
    model,
    tokenizer: spm.SentencePieceProcessor,
    test_examples: list[dict],
    sep_text: str,
    device: str,
    max_new_tokens: int = 8,
    n_example_samples: int = 12,
) -> ReasoningEvalResult:
    """Greedy-decode an answer for every test example and score exact match against
    ``example["answer"]``.

    ``max_new_tokens=8`` is deliberately small -- every ground-truth answer in this
    dataset is one entity name (1-3 sentencepiece tokens, see
    ``reasoning_gen.py``'s entity lists), so an answer that runs past 8 generated
    tokens without hitting EOS has already failed to produce a short, well-formed
    answer regardless of content, and capping generation length here keeps
    evaluation fast across 800 test examples without materially changing which
    examples are scored correct.
    """
    model.eval()
    eos_id = tokenizer.eos_id() if tokenizer.eos_id() >= 0 else None

    n_correct = 0
    kind_correct: dict[str, int] = {}
    kind_total: dict[str, int] = {}
    sampled_examples: list[dict] = []

    for ex in test_examples:
        prompt = f"{ex['prompt']} {sep_text}"
        ids = tokenizer.encode(prompt, out_type=int)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            context = input_ids[:, -model.max_seq_len:]
            logits = model(context)
            next_id = torch.argmax(logits[0, -1, :]).view(1, 1)
            if eos_id is not None and next_id.item() == eos_id:
                break
            input_ids = torch.cat([input_ids, next_id], dim=1)

        generated_ids = input_ids[0, len(ids):].tolist()
        predicted = _normalize(tokenizer.decode(generated_ids))
        gold = _normalize(ex["answer"])
        correct = predicted == gold

        kind = ex["kind"]
        kind_total[kind] = kind_total.get(kind, 0) + 1
        if correct:
            n_correct += 1
            kind_correct[kind] = kind_correct.get(kind, 0) + 1

        if len(sampled_examples) < n_example_samples:
            sampled_examples.append(
                {"prompt": ex["prompt"], "gold": gold, "predicted": predicted, "correct": correct, "kind": kind}
            )

    accuracy_by_kind = {k: kind_correct.get(k, 0) / v for k, v in kind_total.items()}
    return ReasoningEvalResult(
        n_examples=len(test_examples),
        n_correct=n_correct,
        accuracy=n_correct / max(1, len(test_examples)),
        accuracy_by_kind=accuracy_by_kind,
        n_by_kind=kind_total,
        examples=sampled_examples,
    )


def compare_pretrained_vs_finetuned(
    pretrained_checkpoint: Path,
    finetuned_checkpoint: Path,
    tokenizer_path: Path,
    test_split_path: Path,
    sep_text: str,
    load_checkpoint_fn,
    device: str | None = None,
) -> dict:
    """Load both checkpoints, evaluate each on the same test set, return both
    results plus a checkpoint-identity note so the report can cite exactly which
    files produced these numbers.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    test_examples = load_examples(test_split_path)

    results = {}
    for label, ckpt_path in (("pretrained", pretrained_checkpoint), ("finetuned", finetuned_checkpoint)):
        model, _payload, _cfg, _state = load_checkpoint_fn(ckpt_path, device=device)
        model.to(device)
        model.eval()
        result = evaluate_reasoning_accuracy(model, tokenizer, test_examples, sep_text, device)
        results[label] = asdict(result)
        results[label]["checkpoint"] = str(ckpt_path)
        del model  # free before loading the next checkpoint -- both are ~300MB, matters on this machine's RAM budget

    return results


def write_results(results: dict, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
