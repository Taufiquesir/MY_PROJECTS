# Model H (Hindi) pretraining on Kaggle. Paste each `# %%`-delimited block into its
# own notebook cell (most Kaggle/Jupyter-aware editors split on this marker
# automatically -- if not, just copy each block manually).
#
# Setup on Kaggle, once per session:
#   1. Add this repo's GitHub as a notebook "Add-on" or clone it in a cell (below).
#   2. Add your "lma-hindi-dataset" Kaggle Dataset (the Phase 1 corpus mirror) as an
#      input -- it will appear read-only under /kaggle/input/lma-hindi-dataset/.
#   3. If resuming: add your previous checkpoints Kaggle Dataset as a second input,
#      e.g. /kaggle/input/lma-hindi-checkpoints/ -- copy it into
#      /kaggle/working/checkpoints/hindi/ before training starts so `train()` sees
#      latest.pt and resumes instead of starting fresh.
#   4. Enable a GPU accelerator (Settings -> Accelerator -> GPU T4 x2 or P100).
#   5. At the end of the session (or periodically): "Save Version" with
#      /kaggle/working/checkpoints/ included as output, then create/update a
#      Kaggle Dataset from that output notebook version so the next session can
#      mount it as input and resume.

# %% Clone the repo (skip if you added it as a Kaggle Add-on / Github integration instead)
import subprocess

subprocess.run(
    ["git", "clone", "--branch", "phase-2", "https://github.com/CL3-410/individual-project-Taufiquesir.git", "/kaggle/working/repo"],
    check=True,
)

# %% Install the one dependency Kaggle's base image doesn't already have (torch/numpy/pyyaml are preinstalled)
subprocess.run(["pip", "install", "-q", "sentencepiece"], check=True)

# %% Resume setup: copy a previous checkpoints dataset into /kaggle/working/ if present.
# Change CHECKPOINT_INPUT_DIR to match whatever you named your checkpoints dataset,
# or leave it None to start fresh (e.g. on the very first session).
import shutil
from pathlib import Path

CHECKPOINT_INPUT_DIR = None  # e.g. Path("/kaggle/input/lma-hindi-checkpoints/hindi")
CHECKPOINT_OUTPUT_DIR = Path("/kaggle/working/checkpoints/hindi")

if CHECKPOINT_INPUT_DIR is not None and Path(CHECKPOINT_INPUT_DIR).exists():
    CHECKPOINT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in Path(CHECKPOINT_INPUT_DIR).glob("*.pt"):
        shutil.copy(f, CHECKPOINT_OUTPUT_DIR / f.name)
    print(f"copied {list(CHECKPOINT_OUTPUT_DIR.glob('*.pt'))} for resume")
else:
    print("no checkpoint input configured -- starting fresh")

# %% Run training.
# The Kaggle Dataset (taufiquehussain/lma-hindi-dataset) mounts at
# /kaggle/input/lma-hindi-dataset/, but Kaggle nests the actual uploaded folder one
# level deeper under its own name (e.g. .../lma-hindi-dataset/LMA-Phase1-Hindi/),
# so splits/ is not directly at the dataset mount root -- found by inspecting the
# dataset's Data Explorer on kaggle.com before writing this. Auto-detect that one
# nested folder (glob for a `splits` dir up to 2 levels down) rather than
# hardcoding "LMA-Phase1-Hindi", since that inner folder name could change on a
# future dataset version without the outer slug changing.
import sys

sys.path.insert(0, "/kaggle/working/repo")
from hindi.model.config import ModelConfig
from hindi.model.train_loop import TrainConfig, train
from hindi.train.train import prepare_binaries

REPO = Path("/kaggle/working/repo")
DATASET_ROOT = Path("/kaggle/input/lma-hindi-dataset")
SPLITS_DIR = next(DATASET_ROOT.glob("**/splits"))
print(f"found splits dir: {SPLITS_DIR}")
TOKENIZER_PATH = REPO / "hindi" / "tokenizer" / "hindi_tokenizer.model"
BIN_DIR = Path("/kaggle/working/tokenized/hindi")

model_cfg = ModelConfig.load(REPO / "hindi" / "configs" / "model.yaml")
train_cfg = TrainConfig.load(REPO / "hindi" / "configs" / "train.yaml")
print(f"model: {model_cfg}")
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
# /kaggle/working/checkpoints/hindi/{latest.pt, best.pt} is what you version as a
# new Kaggle Dataset (or update an existing one) to resume from next session.
for f in CHECKPOINT_OUTPUT_DIR.glob("*.pt"):
    print(f, f.stat().st_size / 1e6, "MB")
