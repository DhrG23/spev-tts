#!/bin/bash
# Train on expressive/emotional speech data (ESD or Jenny) instead of, or
# blended with, plain LJSpeech - required if you want breathiness/roughness/
# emotion to be *learned* rather than applied heuristically at inference.
#
# Usage:
#   ./train_expressive.sh esd   /path/to/ESD_English
#   ./train_expressive.sh jenny /path/to/Jenny

source venv/bin/activate

DATASET="$1"
IN_DIR="$2"
OUT_DIR="data/training_data_${DATASET}"

if [ -z "$DATASET" ] || [ -z "$IN_DIR" ]; then
    echo "Usage: ./train_expressive.sh <esd|jenny> <path_to_raw_dataset>"
    exit 1
fi

python3 advanced__download_dataset.py \
    --dataset "$DATASET" \
    --in_dir "$IN_DIR" \
    --out_dir "$OUT_DIR"

python3 spev_real_metrics.py \
    --mode train \
    --data_dir "$OUT_DIR" \
    --hifigan_dir vocoder_checkpoints/LJ_FT_T2_V3 \
    --name "run_${DATASET}" \
    --epochs 150

echo ""
echo "Training complete! Checkpoint saved to checkpoints/run_${DATASET}/best.pt"
