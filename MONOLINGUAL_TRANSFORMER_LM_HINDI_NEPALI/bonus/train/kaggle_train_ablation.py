# Bonus ablation (no positional embeddings) pretraining for Model H on Kaggle.
# Paste each `# %%`-delimited block into its own notebook cell.
#
# This repository is private, so Kaggle's `git clone` cannot reach it directly --
# instead, upload the whole `bonus/` folder plus the pieces of `hindi/` and
# `collection/` it depends on (see bonus/README.md for the exact file list) as a
# Kaggle Dataset, and this script copies that uploaded package into a writable
# location instead of cloning.
#
# Setup on Kaggle, once per session:
#   1. Upload the code package (see bonus/README.md) as a Kaggle Dataset, e.g.
#      named "lma-bonus-ablation-code" -- Kaggle mounts it read-only under
#      /kaggle/input/lma-bonus-ablation-code/.
#   2. Add your existing Hindi corpus Kaggle Dataset as a second input -- same one
#      the standard Hindi pretraining run used.
#   3. If resuming: add a third input dataset with your previous ablation
#      checkpoints and copy it in (cell below).
#   4. Settings -> Accelerator -> GPU.
#   5. Run every cell in order.
#   6. Before the session ends: "Save Version" with
#      /kaggle/working/checkpoints/ included as output, then version a Kaggle
#      Dataset from that output so the next session can resume.

# %% Copy the uploaded code package to a writable location and install the one
# missing dependency (torch/numpy/pyyaml are preinstalled on Kaggle's base image).
import shutil
import subprocess
from pathlib import Path

CODE_INPUT_DIR = Path("/kaggle/input/lma-bonus-ablation-code")  # rename to match whatever you name the uploaded dataset
REPO = Path("/kaggle/working/repo")

if REPO.exists():
    shutil.rmtree(REPO)
shutil.copytree(CODE_INPUT_DIR, REPO)
subprocess.run(["pip", "install", "-q", "sentencepiece"], check=True)
print("code copied to", REPO)

# %% Resume setup: copy a previous ablation-checkpoints dataset into /kaggle/working/ if present.
CHECKPOINT_INPUT_DIR = None  # e.g. Path("/kaggle/input/lma-bonus-checkpoints/checkpoints")
CHECKPOINT_OUTPUT_DIR = Path("/kaggle/working/checkpoints/bonus")

if CHECKPOINT_INPUT_DIR is not None and Path(CHECKPOINT_INPUT_DIR).exists():
    CHECKPOINT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in Path(CHECKPOINT_INPUT_DIR).glob("*.pt"):
        shutil.copy(f, CHECKPOINT_OUTPUT_DIR / f.name)
    print(f"copied {list(CHECKPOINT_OUTPUT_DIR.glob('*.pt'))} for resume")
else:
    print("no checkpoint input configured -- starting fresh")

# %% Run training.
# Kaggle nests the uploaded corpus folder one level under the dataset slug (same
# quirk as the standard Hindi run), so this globs for a `splits` directory rather
# than hardcoding the inner folder name.
import sys

sys.path.insert(0, str(REPO))
from hindi.model.config import ModelConfig
from hindi.model.train_loop import TrainConfig
from bonus.model.train_loop_ablation import train
from hindi.train.train import prepare_binaries

DATASET_ROOT = Path("/kaggle/input/lma-phase-hindi-data")  # rename to match your dataset slug if different
SPLITS_DIR = next(DATASET_ROOT.glob("**/splits"))
print(f"found splits dir: {SPLITS_DIR}")
TOKENIZER_PATH = REPO / "hindi" / "tokenizer" / "hindi_tokenizer.model"
BIN_DIR = Path("/kaggle/working/tokenized_ablation/hindi")

model_cfg = ModelConfig.load(REPO / "bonus" / "configs" / "model_ablation.yaml")
train_cfg = TrainConfig.load(REPO / "bonus" / "configs" / "train_ablation.yaml")
print(f"model (ablation, no positional embeddings): {model_cfg}")
print(f"train: {train_cfg}")

train_bin, val_bin = prepare_binaries(SPLITS_DIR, TOKENIZER_PATH, BIN_DIR)

train(
    model_cfg=model_cfg,
    train_cfg=train_cfg,
    train_bin=train_bin,
    val_bin=val_bin,
    checkpoint_dir=CHECKPOINT_OUTPUT_DIR,
    resume=True,  # picks up latest.pt if the resume-copy cell above found one
)

# %% After training (or if the session is about to time out): confirm what got saved.
for f in CHECKPOINT_OUTPUT_DIR.glob("*.pt"):
    print(f, f.stat().st_size / 1e6, "MB")
