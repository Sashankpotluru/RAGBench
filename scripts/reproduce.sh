#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
python scripts/download_data.py

# Retain the original positional retrieval-only invocation.
if [ "$#" -eq 1 ]; then
    python -m benchmarks.run_retrieval --output "$1"
    python -m benchmarks.crosscheck "$1"
    exit 0
fi
if [ "$#" -eq 0 ]; then
    results=reproduced
elif [ "$#" -eq 2 ] && [ "$1" = "--results" ]; then
    results=$2
else
    echo "Usage: $0 [--results new-release-directory | new-retrieval-directory]" >&2
    exit 2
fi
python scripts/download_data.py --model google/flan-t5-small
python -m benchmarks.run_retrieval --output "$results/retrieval"
python -m benchmarks.run_answers --retrieval "$results/retrieval" --output "$results/answers"
python -m benchmarks.release --results "$results"
python -m benchmarks.verify --results "$results"
