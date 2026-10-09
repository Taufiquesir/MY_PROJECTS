# Model L (Nepali) pretraining on Kaggle. Paste each `# %%`-delimited block into its
# own notebook cell. See hindi/train/kaggle_train.py for the fully-commented version
# of this setup -- this file mirrors it exactly except for language-specific paths.

# %% Clone the repo
import subprocess

subprocess.run(
    ["git", "clone", "--branch", "phase-2", "https://github.com/CL3-410/individual-project-Taufiquesir.git", "/kaggle/working/repo"],
    check=True,
)

# %% Install the one dependency Kaggle's base image doesn't already have
subprocess.run(["pip", "install", "-q", "sentencepiece"], check=True)

# %% Resume setup
import shutil
from pathlib import Path

CHECKPOINT_INPUT_DIR = None  # e.g. Path("/kaggle/input/lma-nepali-checkpoints/nepali")
CHECKPOINT_OUTPUT_DIR = Path("/kaggle/working/checkpoints/nepali")

if CHECKPOINT_INPUT_DIR is not None and Path(CHECKPOINT_INPUT_DIR).exists():
    CHECKPOINT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in Path(CHECKPOINT_INPUT_DIR).glob("*.pt"):
        shutil.copy(f, CHECKPOINT_OUTPUT_DIR / f.name)
    print(f"copied {list(CHECKPOINT_OUTPUT_DIR.glob('*.pt'))} for resume")
else:
    print("no checkpoint input configured -- starting fresh")

# %% Run training.
# The Kaggle Dataset (taufiquehussain/lma-nepali-dataset) mounts at
# /kaggle/input/lma-nepali-dataset/, but Kaggle nests the actual uploaded folder one
# level deeper under its own name (confirmed on the Hindi dataset's Data Explorer,
# e.g. .../lma-hindi-dataset/LMA-Phase1-Hindi/; the Nepali one is expected to follow
# the same pattern), so splits/ is not directly at the dataset mount root.
# Auto-detect that nested folder (glob for a `splits` dir up to 2 levels down)
# rather than hardcoding the inner folder name.
import sys

sys.path.insert(0, "/kaggle/working/repo")
from nepali.model.config import ModelConfig
from nepali.model.train_loop import TrainConfig, train
from nepali.train.train import prepare_binaries

REPO = Path("/kaggle/working/repo")
DATASET_ROOT = Path("/kaggle/input/lma-nepali-dataset")
SPLITS_DIR = next(DATASET_ROOT.glob("**/splits"))
print(f"found splits dir: {SPLITS_DIR}")
TOKENIZER_PATH = REPO / "nepali" / "tokenizer" / "nepali_tokenizer.model"
BIN_DIR = Path("/kaggle/working/tokenized/nepali")

model_cfg = ModelConfig.load(REPO / "nepali" / "configs" / "model.yaml")
train_cfg = TrainConfig.load(REPO / "nepali" / "configs" / "train.yaml")
print(f"model: {model_cfg}")
print(f"train: {train_cfg}")

train_bin, val_bin = prepare_binaries(SPLITS_DIR, TOKENIZER_PATH, BIN_DIR)

train(
    model_cfg=model_cfg,
    train_cfg=train_cfg,
    train_bin=train_bin,
    val_bin=val_bin,
    checkpoint_dir=CHECKPOINT_OUTPUT_DIR,
    resume=True,
)

# %% Confirm what got saved
for f in CHECKPOINT_OUTPUT_DIR.glob("*.pt"):
    print(f, f.stat().st_size / 1e6, "MB")
