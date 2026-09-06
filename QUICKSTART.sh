#!/bin/bash

# SPEV TTS - Quick Start Script (Linux/macOS)
# This script sets up the complete SPEV TTS environment

set -e  # Exit on error

echo "=============================================="
echo "  SPEV TTS - Quick Start Setup"
echo "=============================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_header() {
    echo -e "${CYAN}$1${NC}"
}

# Check if running as root (not recommended)
if [ "$EUID" -eq 0 ]; then 
    print_warning "Running as root is not recommended. Consider using a regular user account."
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
print_info "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    echo ""
    echo "Installation instructions:"
    echo "  Ubuntu/Debian: sudo apt-get install python3 python3-pip python3-venv"
    echo "  macOS: brew install python3"
    echo "  Arch: yay -S python312"
    echo "  Fedora: sudo dnf install python3 python3-pip"
    exit 1
fi

PYTHON_VERSION=$(python3.12 --version | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python 3.8 or higher is required. You have Python $PYTHON_VERSION"
    exit 1
fi

print_success "Python $PYTHON_VERSION found"

# Check for git
if ! command -v git &> /dev/null; then
    print_error "git is not installed. Please install git first."
    echo ""
    echo "Installation instructions:"
    echo "  Ubuntu/Debian: sudo apt-get install git"
    echo "  macOS: brew install git"
    echo "  Fedora: sudo dnf install git"
    exit 1
fi

# Create virtual environment
print_info "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3.12 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists, skipping..."
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded"

# Install PyTorch (CPU version, change for CUDA if needed)
print_header ""
print_header "========== Installing PyTorch =========="
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detected!"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1
    echo ""
    print_info "Installing PyTorch with CUDA 11.8 support..."
    echo "This may take a few minutes..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
    print_success "PyTorch (CUDA) installed"
else
    print_warning "No GPU detected, installing CPU version..."
    echo "Note: Training will be MUCH slower on CPU (24-48 hours vs 4-6 hours)"
    echo "This may take a few minutes..."
    pip install torch torchaudio
    print_success "PyTorch (CPU) installed"
fi

# Install dependencies
print_header ""
print_header "========== Installing Dependencies =========="
print_info "Installing required Python packages..."
echo "This may take a few minutes..."

# Install packages with progress
pip install librosa soundfile numpy scipy textgrid phonemizer pandas requests tqdm

print_success "All dependencies installed"

# phonemizer needs the espeak-ng system binary, not just the pip package
if ! command -v espeak-ng &> /dev/null; then
    print_warning "espeak-ng not found (required by phonemizer for text->phoneme conversion)."
    echo "  Ubuntu/Debian: sudo apt-get install espeak-ng"
    echo "  macOS: brew install espeak-ng"
else
    print_success "espeak-ng found"
fi

# Create directory structure
print_info "Creating directory structure..."
mkdir -p data/raw_ljspeech
mkdir -p data/training_data_ljspeech
mkdir -p data/textgrid_data
mkdir -p vocoder_checkpoints
mkdir -p checkpoints
mkdir -p output
print_success "Directories created"

# Check if HiFi-GAN repository exists
print_header ""
print_header "========== Setting up HiFi-GAN Vocoder =========="
if [ ! -d "hifi-gan" ]; then
    print_info "Cloning HiFi-GAN repository..."
    git clone https://github.com/jik876/hifi-gan.git
    print_success "HiFi-GAN repository cloned"
else
    print_warning "HiFi-GAN repository already exists, skipping clone..."
fi

# Download HiFi-GAN V3 checkpoint
print_info "Checking HiFi-GAN V3 checkpoint..."
HIFIGAN_DIR="vocoder_checkpoints/LJ_FT_T2_V3"
HIFIGAN_CHECKPOINT="${HIFIGAN_DIR}/generator_v3"
HIFIGAN_CONFIG="${HIFIGAN_DIR}/config.json"
HIFIGAN_CHECKPOINT_URL="https://github.com/csukuangfj/models/raw/refs/heads/master/hifigan/generator_v3"
HIFIGAN_CONFIG_URL="https://raw.githubusercontent.com/jik876/hifi-gan/master/config_v3.json"

mkdir -p "$HIFIGAN_DIR"

if [ ! -f "$HIFIGAN_CHECKPOINT" ]; then
    print_info "Downloading HiFi-GAN V3 generator from maintained GitHub mirror..."
    if command -v curl &> /dev/null; then
        curl -fL --retry 3 --retry-delay 2 "$HIFIGAN_CHECKPOINT_URL" -o "$HIFIGAN_CHECKPOINT"
    elif command -v wget &> /dev/null; then
        wget --tries=3 "$HIFIGAN_CHECKPOINT_URL" -O "$HIFIGAN_CHECKPOINT"
    else
        print_error "Neither curl nor wget is installed."
        exit 1
    fi
    print_success "HiFi-GAN V3 generator downloaded"
else
    print_success "HiFi-GAN V3 generator already exists"
fi

if [ ! -f "$HIFIGAN_CONFIG" ]; then
    print_info "Downloading official HiFi-GAN V3 config..."
    if command -v curl &> /dev/null; then
        curl -fL --retry 3 --retry-delay 2 "$HIFIGAN_CONFIG_URL" -o "$HIFIGAN_CONFIG"
    else
        wget --tries=3 "$HIFIGAN_CONFIG_URL" -O "$HIFIGAN_CONFIG"
    fi
    print_success "HiFi-GAN V3 config downloaded"
else
    print_success "HiFi-GAN V3 config already exists"
fi

# Normalize the mirror checkpoint into the standard HiFi-GAN checkpoint format.
# Upstream HiFi-GAN expects torch.load(... )['generator'].
print_info "Validating HiFi-GAN V3 checkpoint format..."
python3 - "$HIFIGAN_CHECKPOINT" <<'PY'
import sys
from pathlib import Path
import torch

checkpoint_path = Path(sys.argv[1])
tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".normalized")
obj = torch.load(checkpoint_path, map_location="cpu")

if isinstance(obj, dict) and "generator" in obj:
    state = obj
else:
    state = {"generator": obj}

torch.save(state, tmp_path)
tmp_path.replace(checkpoint_path)
print("checkpoint format OK")
PY
print_success "HiFi-GAN V3 checkpoint validated"

# Check if MFA aligned cache exists
print_header ""
print_header "========== Checking Training Data =========="
if [ -f "proper_cache_strict.pt" ]; then
    print_success "MFA aligned cache (proper_cache_strict.pt) found!"
    print_info "You can skip dataset download and alignment."
    CACHE_SIZE=$(du -h proper_cache_strict.pt | cut -f1)
    echo "Cache size: $CACHE_SIZE"
else
    print_warning "MFA aligned cache (proper_cache_strict.pt) not found."
    echo ""
    echo "You have three options:"
    echo "  1. Place your pre-existing 'proper_cache_strict.pt' in the project root"
    echo "  2. Download and prepare the dataset (next step will guide you)"
    echo "  3. Train will automatically generate cache on first run (after MFA alignment)"
    echo ""
fi

# Ask user if they want to download dataset
echo ""
read -p "Do you want to download the LJSpeech dataset (~2.6GB)? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_header ""
    print_header "========== Downloading LJSpeech Dataset =========="
    print_info "This will take 10-30 minutes depending on your connection..."
    python3 download_datasets.py --dataset single-speaker
    print_success "Dataset downloaded and prepared"
    
    # Check if conda is available for MFA
    print_header ""
    print_header "========== Montreal Forced Aligner Setup =========="
    echo ""
    if command -v conda &> /dev/null; then
        print_success "Conda found! You can install MFA."
        echo ""
        read -p "Install Montreal Forced Aligner now? (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Installing MFA..."
            conda install -c conda-forge montreal-forced-aligner -y
            print_success "MFA installed"
            
            print_info "Downloading MFA models..."
            mfa model download acoustic english_us_arpa
            mfa model download dictionary english_us_arpa
            print_success "MFA models downloaded"
            
            echo ""
            read -p "Run alignment now? (This takes 30-60 minutes) (y/n): " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                print_info "Starting MFA alignment..."
                echo "This will take 30-60 minutes. Please be patient..."
                mfa align data/training_data_ljspeech english_us_arpa english_us_arpa data/textgrid_data
                print_success "Alignment complete!"
            else
                print_info "Skipping alignment. Run it manually later:"
                echo "  mfa align data/training_data_ljspeech english_us_arpa english_us_arpa data/textgrid_data"
            fi
        fi
    else
        print_warning "Conda not found. MFA requires Anaconda or Miniconda."
        echo ""
        echo "To install MFA:"
        echo "1. Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html"
        echo "2. Run: conda install -c conda-forge montreal-forced-aligner"
        echo "3. Download models:"
        echo "   mfa model download acoustic english_us_arpa"
        echo "   mfa model download dictionary english_us_arpa"
        echo "4. Run alignment:"
        echo "   mfa align data/training_data_ljspeech english_us_arpa english_us_arpa data/textgrid_data"
        echo ""
    fi
else
    print_info "Skipping dataset download."
fi

# Create a simple test script
print_header ""
print_header "========== Creating Helper Scripts =========="
print_info "Creating test inference script..."
cat > test_inference.sh << 'EOF'
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
EOF

chmod +x test_inference.sh
print_success "Test inference script created (test_inference.sh)"

# Create advanced (coordinator layer) inference example
# NOTE: the old --nasality/--valence/--arousal/--dominance/--age/
# --lung_capacity/--word_emphasis flags belonged to spev_advanced.py, which
# no longer exists. The current "advanced" layer is the coordinator scripts
# (spev_embodied_core.py / spev_temporal_policy.py), which take --emotion
# and inline [event] tags instead of individual VAD/physiology knobs.
print_info "Creating advanced (coordinator layer) inference example script..."
cat > test_advanced.sh << 'EOF'
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
EOF

chmod +x test_advanced.sh
print_success "Advanced inference script created (test_advanced.sh)"

# Create training script
print_info "Creating training script..."
cat > train_model.sh << 'EOF'
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
    --name run_stable \
    --epochs 100

echo ""
echo "Training complete! Checkpoint saved to checkpoints/run_stable/best.pt"
EOF

chmod +x train_model.sh
print_success "Training script created (train_model.sh)"

# NOTE: there is no separate "advanced" training script anymore.
# spev_embodied_core.py / spev_temporal_policy.py apply their voice-quality
# and emotion effects at inference time on top of whatever checkpoint you
# trained with spev_real_metrics.py - there's nothing extra to train for
# them by default. What *does* require different training data is learned
# (not heuristic) expressiveness: train spev_real_metrics.py itself on an
# expressive corpus (ESD, Jenny) prepared with advanced__download_dataset.py
# instead of - or blended with - plain LJSpeech. Plain LJSpeech alone will
# not produce learned emotional variation no matter how long you train.
print_info "Creating expressive-data training helper script..."
cat > train_expressive.sh << 'EOF'
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
EOF

chmod +x train_expressive.sh
print_success "Expressive-data training script created (train_expressive.sh)"

# Create system check script
print_info "Creating system check script..."
cat > check_system.sh << 'EOF'
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
EOF

chmod +x check_system.sh
print_success "System check script created (check_system.sh)"

# Summary
echo ""
print_header "=============================================="
print_header "  Setup Complete!"
print_header "=============================================="
echo ""
print_success "Environment setup finished successfully!"
echo ""

# Run system check
echo ""
print_info "Running system check..."
./check_system.sh

echo ""
print_header "📋 Next Steps:"
echo ""

if [ -f "proper_cache_strict.pt" ]; then
    echo "1. ✓ MFA cache found - Ready to train!"
    echo ""
    echo -e "   ${GREEN}./train_model.sh${NC}  (Standard FastSpeech 2 on LJSpeech)"
    echo -e "   ${GREEN}./train_expressive.sh esd|jenny <path>${NC}  (Learned emotion/expressiveness)"
else
    echo "1. ⚠ Prepare training data:"
    echo ""
    if [ -d "data/training_data_ljspeech" ] && [ ! -d "data/textgrid_data" ]; then
        echo "   Dataset downloaded. Now run MFA alignment:"
        echo -e "   ${YELLOW}mfa align data/training_data_ljspeech english_us_arpa english_us_arpa data/textgrid_data${NC}"
    elif [ ! -d "data/training_data_ljspeech" ]; then
        echo "   a. Download dataset:"
        echo -e "      ${YELLOW}python3 download_datasets.py --dataset single-speaker${NC}"
        echo ""
        echo "   b. Install MFA (requires conda):"
        echo -e "      ${YELLOW}conda install -c conda-forge montreal-forced-aligner${NC}"
        echo ""
        echo "   c. Download MFA models:"
        echo -e "      ${YELLOW}mfa model download acoustic english_us_arpa${NC}"
        echo -e "      ${YELLOW}mfa model download dictionary english_us_arpa${NC}"
        echo ""
        echo "   d. Run alignment:"
        echo -e "      ${YELLOW}mfa align data/training_data_ljspeech english_us_arpa english_us_arpa data/textgrid_data${NC}"
    fi
    echo ""
    echo "   OR place your pre-existing 'proper_cache_strict.pt' file in the project root"
fi

echo ""
echo "2. After training, test synthesis:"
echo -e "   ${GREEN}./test_inference.sh${NC}  (Standard synthesis)"
echo -e "   ${GREEN}./test_advanced.sh${NC}  (Non-verbal events + emotion curves)"
echo ""
echo "3. Check system status anytime:"
echo -e "   ${GREEN}./check_system.sh${NC}"
echo ""
print_header "📚 Documentation:"
echo "   - README.md - Complete documentation"
echo "   - PRODUCTION_SYSTEM_SUMMARY.md - System architecture and deployment"
echo ""
print_info "Remember to activate the virtual environment before running Python commands:"
echo -e "   ${CYAN}source venv/bin/activate${NC}"
echo ""
print_success "Happy synthesizing! 🎤✨"
echo ""