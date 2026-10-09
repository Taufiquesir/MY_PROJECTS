#!/usr/bin/env bash
# Packages this repo's ablation code and uploads it to Kaggle as a Dataset, using
# the Kaggle CLI. Run this yourself, from your own terminal, on the `bonus`
# branch -- it never touches or needs to touch any AI chat session, since the
# Kaggle API key lives only in your local ~/.kaggle/kaggle.json (standard Kaggle
# CLI convention), never pasted into this script or committed to git.
#
# One-time setup, if you haven't already:
#   pip install kaggle
#
# Auth: `kaggle datasets create`/`version` (upload endpoints) require the newer
# token-based auth, not the legacy username/key ~/.kaggle/kaggle.json pair --
# export your Kaggle API token (the "KGAT_..." string from
# https://www.kaggle.com/settings/api) as an environment variable before running
# this script, e.g.:
#   export KAGGLE_API_TOKEN="KGAT_..."
# (`kaggle datasets list` works fine with either auth method; only the upload
# commands this script actually calls needed the token specifically.)
#
# Usage (from the repo root, on the bonus branch):
#   bash bonus/train/kaggle_upload.sh
#
# First run creates a new Kaggle Dataset named "lma-bonus-ablation-code" under
# your account. Re-running the script after further code changes on this branch
# pushes a new *version* of that same dataset instead of creating a duplicate.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE_DIR="$(mktemp -d)"
DATASET_SLUG="lma-bonus-ablation-code"

echo "staging package at $STAGE_DIR"

mkdir -p "$STAGE_DIR/bonus" "$STAGE_DIR/hindi/model" "$STAGE_DIR/hindi/train" "$STAGE_DIR/hindi/tokenizer" \
         "$STAGE_DIR/collection/common"

# The whole bonus/ folder (code + configs + tests) -- checkpoints/report are
# gitignored *.pt / empty except .gitkeep, so this copies cleanly without
# accidentally including any large binaries even if some exist locally.
cp -r "$REPO_ROOT/bonus/." "$STAGE_DIR/bonus/"
find "$STAGE_DIR/bonus" -name "*.pt" -delete
find "$STAGE_DIR/bonus" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# The pieces of hindi/ and collection/ the ablation depends on -- see
# bonus/README.md's "How to run > On Kaggle" section for why exactly this list.
cp "$REPO_ROOT/hindi/__init__.py" "$STAGE_DIR/hindi/"
cp "$REPO_ROOT/hindi/model/__init__.py" "$REPO_ROOT/hindi/model/config.py" \
   "$REPO_ROOT/hindi/model/data.py" "$REPO_ROOT/hindi/model/transformer.py" \
   "$REPO_ROOT/hindi/model/checkpoint.py" "$REPO_ROOT/hindi/model/train_loop.py" \
   "$STAGE_DIR/hindi/model/"
cp "$REPO_ROOT/hindi/train/__init__.py" "$REPO_ROOT/hindi/train/train.py" "$STAGE_DIR/hindi/train/"
cp "$REPO_ROOT/hindi/tokenizer/hindi_tokenizer.model" "$REPO_ROOT/hindi/tokenizer/hindi_tokenizer.vocab" \
   "$STAGE_DIR/hindi/tokenizer/"
cp "$REPO_ROOT/collection/__init__.py" "$STAGE_DIR/collection/"
cp "$REPO_ROOT/collection/common/__init__.py" "$REPO_ROOT/collection/common/record.py" \
   "$STAGE_DIR/collection/common/"

find "$STAGE_DIR" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

cat > "$STAGE_DIR/dataset-metadata.json" <<EOF
{
  "title": "LMA Bonus Ablation Code",
  "id": "${KAGGLE_USERNAME:-mdtaufiquehussain}/${DATASET_SLUG}",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

echo "package contents:"
find "$STAGE_DIR" -type f | sort

if kaggle datasets list -m --mine 2>/dev/null | grep -q "$DATASET_SLUG"; then
  echo ""
  echo "dataset already exists -- pushing a new version"
  kaggle datasets version -p "$STAGE_DIR" -m "update ablation code" -r zip
else
  echo ""
  echo "creating new dataset"
  kaggle datasets create -p "$STAGE_DIR" -r zip
fi

echo ""
echo "done. attach this dataset to your Kaggle notebook as an input, alongside"
echo "your existing Hindi corpus dataset, then follow bonus/train/kaggle_train_ablation.py."

rm -rf "$STAGE_DIR"
