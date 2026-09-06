#!/bin/bash
# Simple test inference script

source venv/bin/activate

echo "Running test inference..."

CHECKPOINT="checkpoints/run_stable/best.pt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: No checkpoint found at $CHECKPOINT"
    echo "Please train the model first (./train_model.sh), or edit CHECKPOINT"
    echo "above to point at a different run's checkpoint."
    echo ""
    echo "Available checkpoints:"
    find checkpoints -name "*.pt" 2>/dev/null || echo "  No checkpoints found"
    exit 1
fi

python3 spev_real_metrics.py \
    --mode infer \
    --checkpoint "$CHECKPOINT" \
    --text "Hello world! This is a test of the SPEV text to speech system." \
    --duration_scale 1.0 \
    --pitch_scale 1.0 \
    --output output.wav

echo ""
echo "Test complete! Check output.wav"
