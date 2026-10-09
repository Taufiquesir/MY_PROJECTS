"""Supervised (prompt, answer) dataset for reasoning finetuning.

Different shape from ``model/data.py``'s pretraining data pipeline by necessity:
pretraining packs one long, undifferentiated token stream and samples random fixed
windows from it (there is no "the right answer starts here" boundary to respect).
Finetuning examples are short, structured, and the loss must only be computed on the
*answer* tokens -- if the prompt tokens (the given facts and the question) also
contributed to the loss, the model would be partly trained to predict its own given
facts back, which is not the task, and would dilute the gradient signal on the part
that's actually being taught (comparison -> answer).

Each example is tokenized as: ``<bos> <prompt tokens> <sep> <answer tokens> <eos>``,
right-padded to a fixed length with ``pad_id`` (this tokenizer has no dedicated PAD
token -- eos_id is reused for padding, which is safe here because the loss mask
already excludes every position after the true eos). The loss mask is 0 over
``<bos>..<sep>`` inclusive and over padding, 1 over the answer tokens and the
final ``<eos>``.
"""

from __future__ import annotations

import json
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import Dataset

#: Ignore index PyTorch's cross_entropy skips when computing the loss -- standard
#: convention (matches nn.CrossEntropyLoss's default ignore_index of -100).
IGNORE_INDEX = -100


def load_examples(path: Path) -> list[dict]:
    """Read one reasoning-dataset split (JSONL) into a list of dicts."""
    examples = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


class ReasoningDataset(Dataset):
    """Tokenizes (prompt, answer) pairs into fixed-length, loss-masked windows.

    ``sep_text`` is appended after the prompt as a plain-text cue ("उत्तर:" /
    "जवाफ:") rather than a dedicated special token -- this tokenizer's vocabulary
    (frozen, inherited unchanged from Phase 1/2 pretraining per the assignment's
    "keep tokenizer/vocab fixed" finetuning protocol) has no unused id to repurpose
    as a new special token, and introducing one would violate that protocol. A plain
    word the pretrained model has already seen thousands of times in Phase 1 news
    text works as a legible answer-boundary cue without touching the vocabulary.
    """

    def __init__(
        self,
        examples: list[dict],
        tokenizer: spm.SentencePieceProcessor,
        max_seq_len: int,
        sep_text: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.sep_text = sep_text
        self.pad_id = tokenizer.eos_id() if tokenizer.eos_id() >= 0 else 0
        self.examples: list[dict] = []
        n_dropped = 0
        for ex in examples:
            encoded = self._encode(ex["prompt"], ex["answer"])
            if encoded is None:
                n_dropped += 1
                continue
            self.examples.append(encoded)
        self.n_dropped = n_dropped

    def _encode(self, prompt: str, answer: str) -> dict | None:
        bos = self.tokenizer.bos_id()
        eos = self.tokenizer.eos_id()
        prompt_ids = self.tokenizer.encode(f"{prompt} {self.sep_text}", out_type=int)
        answer_ids = self.tokenizer.encode(answer, out_type=int)

        input_ids = [bos] + prompt_ids + answer_ids + [eos]
        # loss mask: 0 over bos+prompt (the model is not being taught to predict
        # facts/questions it was given), 1 over answer+eos.
        loss_mask = [0] * (1 + len(prompt_ids)) + [1] * (len(answer_ids) + 1)

        if len(input_ids) > self.max_seq_len:
            return None  # drop rather than truncate: a truncated answer would silently corrupt the label

        pad_len = self.max_seq_len - len(input_ids)
        input_ids = input_ids + [self.pad_id] * pad_len
        loss_mask = loss_mask + [0] * pad_len
        return {"input_ids": input_ids, "loss_mask": loss_mask}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]
        ids = torch.tensor(ex["input_ids"], dtype=torch.long)
        mask = torch.tensor(ex["loss_mask"], dtype=torch.bool)
        # Standard next-token shift: position t of input predicts token at t+1.
        input_ids = ids[:-1]
        target_ids = ids[1:].clone()
        target_mask = mask[1:]
        target_ids[~target_mask] = IGNORE_INDEX
        return input_ids, target_ids, target_mask
