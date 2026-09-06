#!/bin/bash
# Training script for SPEV TTS

source venv/bin/activate

echo "Starting SPEV TTS training..."
echo "This will take 4-6 hours on GPU, 24-48 hours on CPU"
echo ""

python3 spev_real_metrics.py \
  --mode train \
  --data_dir data/training_data_ljspeech \
  --textgrid_dir data/textgrid_data \
  --hifigan_dir vocoder_checkpoints/LJ_FT_T2_V3 \
  --epochs 100 \
  --batch_size 8 \
  --name run_stable

echo ""
echo "Training complete! Checkpoint saved to checkpoints/run_stable/best.pt"
