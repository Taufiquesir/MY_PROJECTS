"""Generate Model H's (Hindi) synthetic comparative-reasoning finetuning dataset.

Run with: python -m hindi.reasoning.generate

Renders the same comparison/ranking/transitive-chaining ideas as
``nepali/reasoning/generate.py`` but in natural Hindi sentences with Hindi entity
names -- the two languages never share an entity pool, an attribute list, or a
sentence template, only the language-agnostic generation *logic* in
``collection/common/reasoning_gen.py``, so this satisfies "do not share finetuning
data or mix languages" while still being the same reasoning task in spirit.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collection.common.reasoning_gen import Attribute, LanguagePack, build_dataset, write_splits

ENTITIES = [
    "राम", "श्याम", "मोहन", "सोहन", "गीता", "सीता", "राधा", "मीरा",
    "अजय", "विजय", "संजय", "अनिल", "सुनील", "कमल", "विमल", "निर्मल",
    "रोहन", "सोहन", "अर्जुन", "करण", "प्रिया", "पूजा", "नेहा", "काजल",
    "अमन", "रमन", "सुमन", "चमन", "दीपक", "प्रकाश", "आकाश", "विकास",
]
# rohan/sohan collide once dedup'd by set(); acceptable, list stays intentionally
# large and slightly redundant so entity-pool splitting still has enough names.
ENTITIES = sorted(set(ENTITIES))

ATTRIBUTES = [
    Attribute(key="height", noun="ऊँचाई", unit="सेंटीमीटर", lo=140, hi=200),
    Attribute(key="age", noun="उम्र", unit="साल", lo=10, hi=80),
    Attribute(key="price", noun="कीमत", unit="रुपये", lo=100, hi=50000),
    Attribute(key="weight", noun="वज़न", unit="किलोग्राम", lo=30, hi=120),
    Attribute(key="distance", noun="दूरी", unit="किलोमीटर", lo=1, hi=500),  # held out from train
]


def _fact_sentence(entity: str, attr: Attribute, value: int) -> str:
    return f"{entity} की {attr.noun} {value} {attr.unit} है।"


def _pairwise_question(a: str, b: str, attr: Attribute) -> str:
    return f"{a} और {b} में से किसकी {attr.noun} ज़्यादा है?"


def _ranking_question_largest(entities: list[str], attr: Attribute) -> str:
    names = ", ".join(entities[:-1]) + f" और {entities[-1]}"
    return f"{names} में से किसकी {attr.noun} सबसे ज़्यादा है?"


def _ranking_question_smallest(entities: list[str], attr: Attribute) -> str:
    names = ", ".join(entities[:-1]) + f" और {entities[-1]}"
    return f"{names} में से किसकी {attr.noun} सबसे कम है?"


def _transitive_question(hi_e: str, lo_e: str) -> str:
    return f"{hi_e} और {lo_e} में से कौन बड़ा है?"


HINDI_PACK = LanguagePack(
    lang="hindi",
    entities=ENTITIES,
    attributes=ATTRIBUTES,
    fact_sentence=_fact_sentence,
    pairwise_question=_pairwise_question,
    ranking_question_largest=_ranking_question_largest,
    ranking_question_smallest=_ranking_question_smallest,
    transitive_question=_transitive_question,
    larger_word="ज़्यादा",
    smaller_word="कम",
    equal_word="बराबर",
)


def main() -> None:
    splits = build_dataset(HINDI_PACK, n_train=6000, n_val=600, n_test=800, seed=42)
    out_dir = Path(__file__).resolve().parent / "splits"
    manifest = write_splits(splits, out_dir)
    print(f"wrote Hindi reasoning dataset to {out_dir}")
    for split_name, info in manifest["splits"].items():
        print(f"  {split_name}: {info['n_examples']} examples, kinds={info['kind_counts']}")


if __name__ == "__main__":
    main()
