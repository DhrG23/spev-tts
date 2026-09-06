#!/bin/bash
# System check script

echo "========== SPEV TTS System Check =========="
echo ""

echo "Python Version:"
python3 --version
echo ""

echo "PyTorch Version:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
echo ""

echo "CUDA Available:"
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python3 -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
    python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi
echo ""

echo "Installed Packages:"
pip list | grep -E "torch|librosa|soundfile|numpy|textgrid|phonemizer"
echo ""

echo "espeak-ng (required by phonemizer):"
if command -v espeak-ng &> /dev/null; then
    espeak-ng --version
else
    echo "✗ Not found - install via apt/brew (see README.md Requirements)"
fi
echo ""

echo "Directory Structure:"
for dir in data checkpoints vocoder_checkpoints output hifi-gan; do
    if [ -d "$dir" ]; then
        echo "✓ $dir/"
    else
        echo "✗ $dir/ (missing)"
    fi
done
echo ""

echo "Cache File:"
if [ -f "proper_cache_strict.pt" ]; then
    SIZE=$(du -h proper_cache_strict.pt | cut -f1)
    echo "✓ proper_cache_strict.pt ($SIZE)"
else
    echo "✗ proper_cache_strict.pt (not found)"
fi
echo ""

echo "HiFi-GAN Vocoder:"
if [ -d "vocoder_checkpoints/LJ_FT_T2_V3" ]; then
    echo "✓ HiFi-GAN checkpoint found"
    if ls vocoder_checkpoints/LJ_FT_T2_V3/g_* >/dev/null 2>&1; then
        ls -lh vocoder_checkpoints/LJ_FT_T2_V3/g_* | head -n 1
    elif [ -f "vocoder_checkpoints/LJ_FT_T2_V3/generator_v3" ]; then
        ls -lh "vocoder_checkpoints/LJ_FT_T2_V3/generator_v3"
    fi
else
    echo "✗ HiFi-GAN checkpoint not found"
fi
echo ""

echo "Training Data:"
if [ -d "data/training_data_ljspeech" ]; then
    COUNT=$(ls -1 data/training_data_ljspeech/*.wav 2>/dev/null | wc -l)
    echo "✓ Training audio files: $COUNT"
else
    echo "✗ Training data not found"
fi
echo ""

echo "TextGrid Alignments:"
if [ -d "data/textgrid_data" ]; then
    COUNT=$(find data/textgrid_data -name "*.TextGrid" 2>/dev/null | wc -l)
    echo "✓ TextGrid files: $COUNT"
else
    echo "✗ TextGrid data not found"
fi
echo ""

echo "Model Checkpoints:"
if [ -d "checkpoints" ]; then
    COUNT=$(find checkpoints -name "*.pt" 2>/dev/null | wc -l)
    if [ "$COUNT" -gt 0 ]; then
        echo "✓ Checkpoints found: $COUNT"
        find checkpoints -name "*.pt" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -n 3 | cut -d' ' -f2-
    else
        echo "○ No checkpoints yet (train model first)"
    fi
else
    echo "○ No checkpoints yet"
fi

echo ""
echo "=========================================="
