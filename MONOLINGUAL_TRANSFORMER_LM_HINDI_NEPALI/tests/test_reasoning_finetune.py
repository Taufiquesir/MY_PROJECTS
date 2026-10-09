"""Tests the Phase 3 finetuning data pipeline and loop: loss masking is correct
(only answer tokens contribute), and a finetuning run started from a pretrained
checkpoint is resumable with the same checkpoint schema pretraining uses.

Uses Hindi's model/checkpoint code and tokenizer as the concrete language under
test -- ``reasoning_finetune.py`` itself is language-agnostic (see its module
docstring), so testing it through one language's real components exercises the
shared logic; Nepali's mirrored script (``nepali/reasoning/finetune.py``) uses the
identical loop, just with Nepali's own checkpoint/tokenizer passed in.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collection.common.reasoning_data import IGNORE_INDEX, ReasoningDataset
from collection.common.reasoning_finetune import FinetuneConfig, finetune
from hindi.model.checkpoint import TrainingState, load_checkpoint, save_checkpoint
from hindi.model.config import ModelConfig
from hindi.model.transformer import DecoderOnlyTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = REPO_ROOT / "hindi" / "tokenizer" / "hindi_tokenizer.model"

pytestmark = pytest.mark.skipif(not TOKENIZER_PATH.exists(), reason="Hindi tokenizer not present in this checkout")


@pytest.fixture(scope="module")
def tokenizer():
    return spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))


def test_loss_mask_covers_only_answer_and_eos(tokenizer):
    examples = [{"prompt": "राम की उम्र 20 साल है।", "answer": "राम"}]
    ds = ReasoningDataset(examples, tokenizer, max_seq_len=64, sep_text="उत्तर:")
    assert len(ds) == 1
    input_ids, target_ids, target_mask = ds[0]

    answer_ids = tokenizer.encode("राम", out_type=int)
    # loss-carrying positions (answer + eos) should equal len(answer_ids) + 1
    assert target_mask.sum().item() == len(answer_ids) + 1
    # every masked-out target position must be IGNORE_INDEX
    assert (target_ids[~target_mask] == IGNORE_INDEX).all()
    # every unmasked target position must be a real (non-ignored) token id
    assert (target_ids[target_mask] != IGNORE_INDEX).all()


def test_overlength_example_is_dropped_not_truncated(tokenizer):
    long_prompt = " ".join(["राम की उम्र बीस साल है।"] * 40)  # far exceeds max_seq_len
    examples = [{"prompt": long_prompt, "answer": "राम"}]
    ds = ReasoningDataset(examples, tokenizer, max_seq_len=16, sep_text="उत्तर:")
    assert len(ds) == 0
    assert ds.n_dropped == 1


def test_finetune_is_resumable_with_full_checkpoint_schema(tokenizer):
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size(), d_model=16, n_layers=1, n_heads=2, d_ff=32, max_seq_len=32, dropout=0.0)
    model = DecoderOnlyTransformer(**cfg.__dict__)

    examples = [
        {"prompt": "राम की उम्र 20 साल है। श्याम की उम्र 30 साल है। कौन बड़ा है?", "answer": "श्याम"},
        {"prompt": "गीता की उम्र 40 साल है। सीता की उम्र 10 साल है। कौन बड़ा है?", "answer": "गीता"},
    ]
    ds = ReasoningDataset(examples, tokenizer, max_seq_len=32, sep_text="उत्तर:")
    assert len(ds) == 2  # both must fit in the tiny max_seq_len for this test to be meaningful

    with tempfile.TemporaryDirectory() as d:
        pretrained_path = Path(d) / "pretrained.pt"
        save_checkpoint(
            pretrained_path, model, torch.optim.AdamW(model.parameters()),
            torch.optim.lr_scheduler.LambdaLR(torch.optim.AdamW(model.parameters()), lambda s: 1.0),
            cfg, TrainingState(step=0, best_val_loss=float("inf")),
        )

        checkpoint_dir = Path(d) / "finetune_ckpt"
        ft_cfg = FinetuneConfig(epochs=1, batch_size=2, warmup_steps=1, log_every=1000)
        finetune(
            pretrained_checkpoint=pretrained_path,
            finetune_cfg=ft_cfg,
            train_dataset=ds,
            val_dataset=ds,
            checkpoint_dir=checkpoint_dir,
            load_checkpoint=load_checkpoint,
            save_checkpoint=save_checkpoint,
            training_state_cls=TrainingState,
            resume=False,
        )

        latest = torch.load(checkpoint_dir / "latest.pt", map_location="cpu", weights_only=False)
        for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "config", "step", "best_val_loss"):
            assert key in latest
        assert latest["step"] == 1  # one epoch completed

        # a second call with resume=True should continue from epoch 1, not restart
        ft_cfg_2 = FinetuneConfig(epochs=2, batch_size=2, warmup_steps=1, log_every=1000)
        history = finetune(
            pretrained_checkpoint=pretrained_path,
            finetune_cfg=ft_cfg_2,
            train_dataset=ds,
            val_dataset=ds,
            checkpoint_dir=checkpoint_dir,
            load_checkpoint=load_checkpoint,
            save_checkpoint=save_checkpoint,
            training_state_cls=TrainingState,
            resume=True,
        )
        assert len(history["train_loss"]) == 2  # epoch 1's history preserved + epoch 2 appended
