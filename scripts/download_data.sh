#!/usr/bin/env bash
# Download the FIFA 22 player dataset from Kaggle.
#
# Prerequisites:
#   1. pip install kaggle
#   2. Place your Kaggle API token at ~/.kaggle/kaggle.json
#      (https://www.kaggle.com/docs/api)
#
# Dataset: Stefano Leone — "FIFA 22 complete player dataset"
# https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data/raw"
mkdir -p "$DATA_DIR"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install with: pip install kaggle"
  exit 1
fi

echo "Downloading FIFA 22 dataset to $DATA_DIR ..."
kaggle datasets download -d stefanoleone992/fifa-22-complete-player-dataset \
  -p "$DATA_DIR" --unzip

echo "Done. Expected file: $DATA_DIR/players_22.csv"
