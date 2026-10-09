"""Synthetic comparative-reasoning dataset generator, shared by both languages.

The assignment requires each model's Phase 3 finetuning set to be generated
programmatically (not downloaded) so ground-truth labels are known exactly, and to
focus on comparative / transitive reasoning: given ordering facts about two or three
entities (height, age, price, quantity, ...), answer a comparison question, including
multi-hop questions that require chaining two given relations (A > B, B > C => A vs C).

Design choices that hold for both languages (only the ``LanguagePack`` -- entity
names, attribute nouns, unit words, and sentence templates -- differs per language):

- **Held-out generalization split, not a random split of the same pool.** A random
  split over one shared entity/attribute pool would let training and test share
  entity names and attribute types; the model could then pass by memorizing which
  name "is usually described as taller" rather than by performing the comparison.
  Instead entity names are partitioned into disjoint TRAIN_ENTITIES / TEST_ENTITIES
  pools up front, and one attribute (out of the language's full attribute list) is
  withheld from training entirely and used only at test time. Every test example is
  built exclusively from held-out entities and includes at least one held-out
  relation pattern (the 3-entity multi-hop template, reserved for val/test -- see
  ``build_dataset``). This is what the assignment's "held-out entity names or
  held-out relation patterns" requirement asks for, and it means a model that merely
  memorized training triples cannot get test examples right by lookup.
- **Numeric ground truth, not template lookup.** Each example's answer is computed
  from the actual sampled numeric values (e.g. real heights in cm), not copied from
  the template -- so grading a generation against ``answer`` is an exact-match check
  against a fact that was genuinely computed, matching how the assignment wants
  accuracy/exact-match reported against known ground truth.
- **Three question families**: pairwise comparison (2 entities, 1 relation given),
  three-entity ranking (3 entities, all pairwise relations given, asks for
  largest/smallest), and multi-hop transitive chaining (2 relations given, e.g.
  A>B and B>C, asks about A vs C -- the relation between A and C is never stated
  directly, so answering it requires chaining, which is exactly the "short
  multi-hop comparisons" example in the spec).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Attribute:
    """One comparable quantity (e.g. height) in one language.

    ``noun`` is the attribute's natural-language name used in sentence templates
    (e.g. Hindi "ऊँचाई"); ``unit`` is its unit word (e.g. "सेंटीमीटर"); ``lo``/``hi``
    bound the sampled integer values so generated facts stay plausible for the
    attribute (a "price" shouldn't sample the same range as an "age").
    """

    key: str
    noun: str
    unit: str
    lo: int
    hi: int


@dataclass(frozen=True)
class LanguagePack:
    """Everything that differs between Hindi and Nepali: vocabulary and phrasing.

    Generation logic in this module is entirely language-agnostic; swapping this
    dataclass is the only thing needed to retarget the generator at a new language,
    which is what keeps the two languages' finetuning data fully independent (no
    shared templates rendered through translation) while sharing zero risk of a
    logic bug diverging between them.
    """

    lang: str
    entities: list[str]
    attributes: list[Attribute]
    # sentence builders: given two (entity, attribute, value) facts, render the
    # natural-language fact plus the question. Kept as callables (not f-string
    # templates baked into this module) so word order/particles can differ freely
    # between Hindi and Nepali without this module needing per-language branches.
    fact_sentence: Callable[[str, Attribute, int], str]
    pairwise_question: Callable[[str, str, Attribute], str]
    ranking_question_largest: Callable[[list[str], Attribute], str]
    ranking_question_smallest: Callable[[list[str], Attribute], str]
    transitive_question: Callable[[str, str], str]
    larger_word: str
    smaller_word: str
    equal_word: str


def _split_entities(entities: list[str], test_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Partition entity names into disjoint train and held-out (test) pools.

    Deterministic given ``seed`` (not Python's global random state) so regenerating
    the dataset with the same seed always reproduces the same held-out split --
    important for the "report ... how you avoid train-test leakage" requirement to
    mean something stable across regenerations.
    """
    rng = random.Random(seed)
    shuffled = list(entities)
    rng.shuffle(shuffled)
    n_test = max(2, int(len(shuffled) * test_fraction))
    return shuffled[n_test:], shuffled[:n_test]


def _make_example(
    ex_id: str,
    split: str,
    kind: str,
    prompt: str,
    answer: str,
) -> dict:
    return {"id": ex_id, "split": split, "kind": kind, "prompt": prompt, "answer": answer}


def _gen_pairwise(pack: LanguagePack, entities: list[str], attr: Attribute, rng: random.Random) -> dict | None:
    a, b = rng.sample(entities, 2)
    va, vb = rng.randint(attr.lo, attr.hi), rng.randint(attr.lo, attr.hi)
    if va == vb:
        return None
    fact1 = pack.fact_sentence(a, attr, va)
    fact2 = pack.fact_sentence(b, attr, vb)
    question = pack.pairwise_question(a, b, attr)
    prompt = f"{fact1} {fact2} {question}"
    answer = a if va > vb else b
    return {"prompt": prompt, "answer": answer, "kind": "pairwise"}


def _gen_ranking(pack: LanguagePack, entities: list[str], attr: Attribute, rng: random.Random) -> dict | None:
    trio = rng.sample(entities, 3)
    values = {e: rng.randint(attr.lo, attr.hi) for e in trio}
    if len(set(values.values())) < 3:
        return None
    facts = " ".join(pack.fact_sentence(e, attr, values[e]) for e in trio)
    ask_largest = rng.random() < 0.5
    question = pack.ranking_question_largest(trio, attr) if ask_largest else pack.ranking_question_smallest(trio, attr)
    answer = max(values, key=values.get) if ask_largest else min(values, key=values.get)
    return {"prompt": f"{facts} {question}", "answer": answer, "kind": "ranking"}


def _gen_transitive(pack: LanguagePack, entities: list[str], attr: Attribute, rng: random.Random) -> dict | None:
    """A > B and B > C given directly; question asks about A vs C, which is never
    stated -- answering requires chaining the two given relations transitively.
    """
    a, b, c = rng.sample(entities, 3)
    va, vb, vc = sorted([rng.randint(attr.lo, attr.hi) for _ in range(3)], reverse=True)
    if va == vb or vb == vc:
        return None
    # Randomize which entity gets which rank so the answer isn't positionally biased.
    hi_e, mid_e, lo_e = rng.sample([a, b, c], 3)
    fact1 = pack.fact_sentence(hi_e, attr, va)
    fact2 = pack.fact_sentence(mid_e, attr, vb)
    fact3 = pack.fact_sentence(lo_e, attr, vc)
    # Only state the two adjacent relations explicitly, as sentences of fact -- the
    # question then asks about the pair whose relation was never stated directly.
    question = pack.transitive_question(hi_e, lo_e)
    prompt = f"{fact1} {fact2} {fact3} {question}"
    return {"prompt": prompt, "answer": hi_e, "kind": "transitive"}


GENERATORS: dict[str, Callable] = {
    "pairwise": _gen_pairwise,
    "ranking": _gen_ranking,
    "transitive": _gen_transitive,
}


def build_dataset(
    pack: LanguagePack,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 42,
    entity_test_fraction: float = 0.25,
    holdout_attribute_fraction: float = 0.2,
) -> dict[str, list[dict]]:
    """Generate train/val/test splits of comparative-reasoning examples.

    Two independent axes of held-out generalization, both required by the
    assignment ("held-out entity names OR held-out relation patterns" -- this
    generator uses both, which is stricter than either alone):

    1. **Held-out entities**: ``entity_test_fraction`` of each language's entity
       name pool is reserved exclusively for val/test (see :func:`_split_entities`).
       Train never sees these names in any example.
    2. **Held-out attribute**: the last ``holdout_attribute_fraction`` of the
       language's attribute list (e.g. one of height/age/price/weight) is used only
       in val/test, never in train -- so test also exercises a comparison *type* the
       model never trained on directly, not just new names for a familiar type.

    The three ``kind`` generators (pairwise, ranking, transitive) are all present in
    every split so no split is missing a question family, but the transitive
    (multi-hop) family is intentionally weighted higher in val/test than in train
    (see the ``kind_weights`` below) since multi-hop chaining is the harder skill the
    assignment specifically calls out and pretrained-vs-finetuned accuracy on it is
    the most informative single number for the Phase 3 report.
    """
    train_entities, test_entities = _split_entities(pack.entities, entity_test_fraction, seed)
    n_holdout_attrs = max(1, int(len(pack.attributes) * holdout_attribute_fraction))
    train_attrs = pack.attributes[:-n_holdout_attrs]
    all_attrs = pack.attributes  # val/test may draw from the full attribute list

    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    targets = {"train": n_train, "val": n_val, "test": n_test}
    kind_weights = {
        "train": {"pairwise": 0.45, "ranking": 0.35, "transitive": 0.20},
        "val": {"pairwise": 0.30, "ranking": 0.30, "transitive": 0.40},
        "test": {"pairwise": 0.30, "ranking": 0.30, "transitive": 0.40},
    }

    for split_name, target in targets.items():
        entities = train_entities if split_name == "train" else test_entities
        attrs = train_attrs if split_name == "train" else all_attrs
        weights = kind_weights[split_name]
        kinds = list(weights.keys())
        probs = [weights[k] for k in kinds]
        seen_prompts: set[str] = set()
        attempts = 0
        max_attempts = target * 20
        while len(splits[split_name]) < target and attempts < max_attempts:
            attempts += 1
            kind = rng.choices(kinds, weights=probs, k=1)[0]
            attr = rng.choice(attrs)
            example = GENERATORS[kind](pack, entities, attr, rng)
            if example is None:
                continue
            if example["prompt"] in seen_prompts:
                continue  # dedup identical prompts (possible with a small entity pool)
            seen_prompts.add(example["prompt"])
            ex_id = hashlib.sha256(f"{pack.lang}:{split_name}:{example['prompt']}".encode("utf-8")).hexdigest()[:16]
            splits[split_name].append(
                _make_example(ex_id, split_name, example["kind"], example["prompt"], example["answer"])
            )

    return splits


def write_splits(splits: dict[str, list[dict]], out_dir: Path) -> dict:
    """Write each split to ``out_dir/<split>.jsonl`` and return a size/kind-mix manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"splits": {}}
    for split_name, examples in splits.items():
        path = out_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        kind_counts: dict[str, int] = {}
        for ex in examples:
            kind_counts[ex["kind"]] = kind_counts.get(ex["kind"], 0) + 1
        manifest["splits"][split_name] = {"n_examples": len(examples), "kind_counts": kind_counts}
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
