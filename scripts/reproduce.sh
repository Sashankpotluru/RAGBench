#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
python scripts/download_data.py
python -m benchmarks.run_retrieval --output "${1:-reproduced/retrieval}"
python -m benchmarks.crosscheck "${1:-reproduced/retrieval}"
