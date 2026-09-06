#!/bin/bash
# Advanced inference: non-verbal events + time-varying emotion curves,
# layered on top of the base checkpoint via the coordinator scripts.

source venv/bin/activate

echo "Running advanced inference (coordinator layer)..."

CHECKPOINT="checkpoints/run_stable/best.pt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: No checkpoint found at $CHECKPOINT"
    echo "Please train the model first (./train_model.sh)."
    exit 1
fi

echo ""
echo "1. spev_embodied_core.py - inline non-verbal events ([sigh], [breath])"
python3 spev_embodied_core.py \
    --text "I am so tired... [sigh] but I must go on." \
    --emotion exhausted \
    --checkpoint "$CHECKPOINT" \
    --hifigan_dir ./hifi-gan \
    --output output_embodied.wav

echo ""
echo "2. spev_temporal_policy.py - time-varying emotion curves"
python3 spev_temporal_policy.py \
    --text "Oh my god, I am so relieved." \
    --emotion relief \
    --checkpoint "$CHECKPOINT" \
    --hifigan_dir ./hifi-gan \
    --output output_temporal.wav

echo ""
echo "Advanced test complete! Check output_embodied.wav and output_temporal.wav"
