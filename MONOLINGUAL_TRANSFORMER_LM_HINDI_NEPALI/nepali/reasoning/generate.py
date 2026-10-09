"""Generate Model L's (Nepali) synthetic comparative-reasoning finetuning dataset.

Run with: python -m nepali.reasoning.generate

See ``hindi/reasoning/generate.py`` for the mirrored Hindi generator -- entity
names, attribute vocabulary, and sentence templates here are entirely independent
(no shared pool, no translation of the Hindi strings), only the generation logic in
``collection/common/reasoning_gen.py`` is shared, matching the assignment's
requirement that finetuning data not be shared or mixed across languages.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collection.common.reasoning_gen import Attribute, LanguagePack, build_dataset, write_splits

ENTITIES = [
    "राम", "हरि", "गोपाल", "विष्णु", "सीता", "गीता", "सुनिता", "अनिता",
    "विनोद", "प्रमोद", "सुरेश", "महेश", "दिनेश", "रमेश", "कमला", "विमला",
    "प्रकाश", "आकाश", "विकास", "उमेश", "पुष्प", "सागर", "बिनोद", "गणेश",
    "पूर्णिमा", "सरिता", "ममता", "रीता", "बिष्णु", "कृष्ण", "अर्जुन", "भीम",
]
ENTITIES = sorted(set(ENTITIES))

ATTRIBUTES = [
    Attribute(key="height", noun="उचाइ", unit="सेन्टिमिटर", lo=140, hi=200),
    Attribute(key="age", noun="उमेर", unit="वर्ष", lo=10, hi=80),
    Attribute(key="price", noun="मूल्य", unit="रुपैयाँ", lo=100, hi=50000),
    Attribute(key="weight", noun="तौल", unit="किलोग्राम", lo=30, hi=120),
    Attribute(key="distance", noun="दूरी", unit="किलोमिटर", lo=1, hi=500),  # held out from train
]


def _fact_sentence(entity: str, attr: Attribute, value: int) -> str:
    return f"{entity}को {attr.noun} {value} {attr.unit} छ।"


def _pairwise_question(a: str, b: str, attr: Attribute) -> str:
    return f"{a} र {b} मध्ये कसको {attr.noun} बढी छ?"


def _ranking_question_largest(entities: list[str], attr: Attribute) -> str:
    names = ", ".join(entities[:-1]) + f" र {entities[-1]}"
    return f"{names} मध्ये कसको {attr.noun} सबैभन्दा बढी छ?"


def _ranking_question_smallest(entities: list[str], attr: Attribute) -> str:
    names = ", ".join(entities[:-1]) + f" र {entities[-1]}"
    return f"{names} मध्ये कसको {attr.noun} सबैभन्दा कम छ?"


def _transitive_question(hi_e: str, lo_e: str) -> str:
    return f"{hi_e} र {lo_e} मध्ये को ठूलो छ?"


NEPALI_PACK = LanguagePack(
    lang="nepali",
    entities=ENTITIES,
    attributes=ATTRIBUTES,
    fact_sentence=_fact_sentence,
    pairwise_question=_pairwise_question,
    ranking_question_largest=_ranking_question_largest,
    ranking_question_smallest=_ranking_question_smallest,
    transitive_question=_transitive_question,
    larger_word="बढी",
    smaller_word="कम",
    equal_word="बराबर",
)


def main() -> None:
    splits = build_dataset(NEPALI_PACK, n_train=6000, n_val=600, n_test=800, seed=42)
    out_dir = Path(__file__).resolve().parent / "splits"
    manifest = write_splits(splits, out_dir)
    print(f"wrote Nepali reasoning dataset to {out_dir}")
    for split_name, info in manifest["splits"].items():
        print(f"  {split_name}: {info['n_examples']} examples, kinds={info['kind_counts']}")


if __name__ == "__main__":
    main()
