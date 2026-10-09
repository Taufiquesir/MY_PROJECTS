"""Decontamination: keep Phase 3 eval text out of the pretraining corpus.

The Phase 3 synthetic reasoning dataset (comparative/ordering questions over
programmatically-generated entities) is built after Phase 1's corpus is collected, so
there is nothing to decontaminate against yet -- this module exists so that step is a
one-line call once that dataset exists, not a retrofit into clean_corpus later.

Two checks, both optional (a no-op if no contamination set is supplied):

1. Exact substring: any held-out eval question/answer text appearing verbatim in a
   training document.
2. Entity leakage: any of the eval set's held-out entity names (the names the
   synthetic-reasoning generator is required to keep out of the training templates,
   per the Phase 3 spec's "held-out entity names" leakage-avoidance requirement)
   appearing in a training document. This is deliberately conservative -- it can flag
   real documents that happen to mention a name reused as a synthetic-eval entity --
   but for a generated dataset the entity list is small and can be chosen from names
   unlikely to appear in real news/encyclopedia text, keeping false positives rare.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ContaminationSet:
    """A loaded decontamination reference: eval snippets and held-out entity names."""

    eval_snippets: set[str] = field(default_factory=set)
    held_out_entities: set[str] = field(default_factory=set)

    @classmethod
    def from_files(cls, eval_text_paths: list[Path], entity_list_path: Path | None = None) -> "ContaminationSet":
        """Load eval text (one snippet per line per file) and an optional entity list."""
        snippets: set[str] = set()
        for path in eval_text_paths:
            if not path.exists():
                continue
            snippets.update(line.strip() for line in path.read_text("utf-8").splitlines() if line.strip())
        entities: set[str] = set()
        if entity_list_path is not None and entity_list_path.exists():
            entities.update(line.strip() for line in entity_list_path.read_text("utf-8").splitlines() if line.strip())
        return cls(eval_snippets=snippets, held_out_entities=entities)

    def is_empty(self) -> bool:
        return not self.eval_snippets and not self.held_out_entities


def is_contaminated(text: str, contamination: ContaminationSet) -> bool:
    """True if ``text`` contains any eval snippet or held-out entity name verbatim."""
    if contamination.is_empty():
        return False
    for snippet in contamination.eval_snippets:
        if snippet in text:
            return True
    for entity in contamination.held_out_entities:
        if entity in text:
            return True
    return False
