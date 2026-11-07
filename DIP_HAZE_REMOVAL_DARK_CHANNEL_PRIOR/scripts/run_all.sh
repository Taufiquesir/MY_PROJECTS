#!/usr/bin/env bash
# Regenerate every reported number and figure. Run from the project root.
# Prereq: data in data/reside/sots/SOTS and data/revide/indoor (see README); paths in configs/data.yaml.
set -euo pipefail
PY=${PY:-.venv/bin/python}
export PYTHONWARNINGS=ignore

$PY -m pytest -q tests                      # synthetic round-trip correctness (Obj. 1)
$PY -m eval.ablation                        # single-image benchmark + OFAT ablation (Obj. 2, 3); resumable
$PY -m eval.revide_runner                   # V0-V3 x alpha sweep on REVIDE test (Obj. 5); resumable
$PY -m eval.flicker_diagnosis               # which cause of flicker dominates (Obj. 4)
$PY -m eval.failure_analysis                # sky failure + mitigation (Obj. 6)
$PY scripts/make_figures.py                 # figures/*.png from results/*.csv
$PY scripts/make_videos.py J004             # side-by-side demo mp4 + transmission maps
