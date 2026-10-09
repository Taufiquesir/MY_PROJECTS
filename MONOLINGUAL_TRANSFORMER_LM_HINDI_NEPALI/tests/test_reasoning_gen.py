"""Pins the Phase 3 synthetic reasoning generator's leakage-avoidance guarantees.

Both properties tested here are the ones the assignment explicitly asks to be
reported ("how you avoid train-test leakage (e.g. held-out entity names or
held-out relation patterns)") -- this test is what makes that report claim
verifiable rather than just asserted.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collection.common.reasoning_gen import Attribute, LanguagePack, build_dataset

TOY_ENTITIES = [f"E{i}" for i in range(20)]
TOY_ATTRS = [
    Attribute(key="a", noun="a", unit="u", lo=1, hi=100),
    Attribute(key="b", noun="b", unit="u", lo=1, hi=100),
    Attribute(key="held_out", noun="c", unit="u", lo=1, hi=100),
]


def _fact(entity: str, attr: Attribute, value: int) -> str:
    return f"{entity} {attr.key} {value}."


def _pair_q(a: str, b: str, attr: Attribute) -> str:
    return f"{a} vs {b} on {attr.key}?"


def _rank_hi(entities: list[str], attr: Attribute) -> str:
    return f"largest of {entities} on {attr.key}?"


def _rank_lo(entities: list[str], attr: Attribute) -> str:
    return f"smallest of {entities} on {attr.key}?"


def _trans_q(a: str, b: str) -> str:
    return f"{a} vs {b}?"


TOY_PACK = LanguagePack(
    lang="toy",
    entities=TOY_ENTITIES,
    attributes=TOY_ATTRS,
    fact_sentence=_fact,
    pairwise_question=_pair_q,
    ranking_question_largest=_rank_hi,
    ranking_question_smallest=_rank_lo,
    transitive_question=_trans_q,
    larger_word="more",
    smaller_word="less",
    equal_word="equal",
)


def _entity_tokens(prompt: str) -> set[str]:
    return {tok for tok in prompt.replace(".", " ").replace("?", " ").split() if tok.startswith("E")}


def test_no_prompt_overlap_between_splits():
    splits = build_dataset(TOY_PACK, n_train=200, n_val=40, n_test=40, seed=1)
    train_prompts = {ex["prompt"] for ex in splits["train"]}
    test_prompts = {ex["prompt"] for ex in splits["test"]}
    val_prompts = {ex["prompt"] for ex in splits["val"]}
    assert train_prompts.isdisjoint(test_prompts)
    assert train_prompts.isdisjoint(val_prompts)


def test_test_entities_never_appear_in_train():
    splits = build_dataset(TOY_PACK, n_train=200, n_val=40, n_test=40, seed=1)
    train_entities: set[str] = set()
    for ex in splits["train"]:
        train_entities |= _entity_tokens(ex["prompt"])
    test_entities: set[str] = set()
    for ex in splits["test"]:
        test_entities |= _entity_tokens(ex["prompt"])
    assert train_entities.isdisjoint(test_entities)
    assert len(test_entities) > 0


def test_holdout_attribute_absent_from_train_present_in_test():
    splits = build_dataset(TOY_PACK, n_train=200, n_val=40, n_test=40, seed=1, holdout_attribute_fraction=0.4)
    train_has_holdout = any("held_out" in ex["prompt"] for ex in splits["train"])
    test_has_holdout = any("held_out" in ex["prompt"] for ex in splits["test"])
    assert not train_has_holdout
    assert test_has_holdout


def test_answers_are_computed_not_copied_from_template():
    splits = build_dataset(TOY_PACK, n_train=50, n_val=10, n_test=10, seed=2)
    for ex in splits["train"] + splits["test"]:
        assert ex["answer"] in _entity_tokens(ex["prompt"])


def test_deterministic_given_seed():
    a = build_dataset(TOY_PACK, n_train=50, n_val=10, n_test=10, seed=7)
    b = build_dataset(TOY_PACK, n_train=50, n_val=10, n_test=10, seed=7)
    assert a == b
