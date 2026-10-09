"""Model configuration: a plain dataclass, serializable to/from YAML, so the exact
hyperparameters used for a given checkpoint are always reconstructable from the
config file saved alongside it (required for checkpoint resume and for reproducing
a run from the README).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    """Architecture hyperparameters for :class:`model.transformer.DecoderOnlyTransformer`.

    ``vocab_size`` is the one field that legitimately differs between Model H and
    Model L (each language's own tokenizer has its own vocabulary) -- both landed at
    9,000 in Phase 1, so in practice the two configs are numerically identical, but
    the field is not hardcoded shared state: each language keeps its own config file.
    """

    vocab_size: int
    d_model: int = 384
    n_layers: int = 13
    n_heads: int = 6
    d_ff: int = 1536
    max_seq_len: int = 256
    dropout: float = 0.1

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ModelConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
