#!/bin/bash

# Majority class guessing over a 5-fold cross validation.

set -euo pipefail

# the repo root is the parent of scripts/, resolved from this file's own location so
# the script can be launched from any working directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# python virtual environment created as described in the README
# change VENV if yours lives somewhere else
VENV=".venv"
source "$VENV/bin/activate"

# specialty: 6 classes, the rare specialties are merged into 'outra'
python3 src/majority.py \
    --path_jsonl data/specialty.jsonl \
    --to_train specialty \
    --num_folds 5 \
    --output_dir ./outputs/majority_specialty

# type: 3 classes, 'admissão de pronto-socorro' is dropped
python3 src/majority.py \
    --path_jsonl data/type.jsonl \
    --to_train type \
    --num_folds 5 \
    --output_dir ./outputs/majority_type
