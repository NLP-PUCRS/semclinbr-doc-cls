#!/bin/bash

# Trains TF-IDF + LinearSVC on both classification objectives, 
# with Optuna tuning (100 trials per fold), 5-fold cross validation and a word cloud per class.

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
python3 src/tfidf.py \
    --path_jsonl data/specialty.jsonl \
    --tune \
    --to_train specialty \
    --trials 100 \
    --tuning_seed 404 \
    --num_folds 5 \
    --output_dir ./outputs/tfidf_specialty \
    --word_clouds

# type: 3 classes, 'admissão de pronto-socorro' is dropped
python3 src/tfidf.py \
    --path_jsonl data/type.jsonl \
    --tune \
    --to_train type \
    --trials 100 \
    --tuning_seed 404 \
    --num_folds 5 \
    --output_dir ./outputs/tfidf_type \
    --word_clouds
