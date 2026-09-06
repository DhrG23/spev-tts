# SPEV TTS - Quick Start Script (Windows PowerShell)
# This script sets up the complete SPEV TTS environment

# Enable strict mode
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  SPEV TTS - Quick Start Setup" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

# Function to print colored messages
function Print-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Print-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor Blue
}

function Print-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

function Print-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

# Check Python version
Print-Info "Checking Python version..."
try {
    $pythonVersion = python --version 2>&1
    Print-Success "Python found: $pythonVersion"
} catch {
    Print-Error "Python is not installed or not in PATH. Please install Python 3.8 or higher."
    exit 1
}

# Create virtual environment
Print-Info "Creating virtual environment..."
if (-not (Test-Path "venv")) {
    python -m venv venv
    Print-Success "Virtual environment created"
} else {
    Print-Warning "Virtual environment already exists, skipping..."
}

# Activate virtual environment
Print-Info "Activating virtual environment..."
& .\venv\Scripts\Activate.ps1
Print-Success "Virtual environment activated"

# Upgrade pip
Print-Info "Upgrading pip..."
python -m pip install --upgrade pip | Out-Null
Print-Success "pip upgraded"

# Detect GPU
Print-Info "Detecting GPU..."
$hasNvidiaGpu = $false
try {
    nvidia-smi | Out-Null
    $hasNvidiaGpu = $true
    Print-Success "NVIDIA GPU detected"
} catch {
    Print-Warning "No NVIDIA GPU detected"
}

# Install PyTorch
Print-Info "Installing PyTorch..."
if ($hasNvidiaGpu) {
    Print-Info "Installing CUDA version of PyTorch..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
} else {
    Print-Warning "Installing CPU version of PyTorch..."
    pip install torch torchaudio
}
Print-Success "PyTorch installed"

# Install dependencies
Print-Info "Installing Python dependencies..."
pip install librosa soundfile numpy scipy textgrid phonemizer pandas requests tqdm | Out-Null
Print-Success "Dependencies installed"

# Create directory structure
Print-Info "Creating directory structure..."
$directories = @(
    "data\raw_ljspeech",
    "data\training_data_ljspeech",
    "data\textgrid_data",
    "vocoder_checkpoints",
    "checkpoints",
    "output"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Print-Success "Directories created"

# Check if HiFi-GAN repository exists
Print-Info "Setting up HiFi-GAN vocoder..."
if (-not (Test-Path "hifi-gan")) {
    Print-Info "Cloning HiFi-GAN repository..."
    git clone https://github.com/jik876/hifi-gan.git
    Print-Success "HiFi-GAN repository cloned"
} else {
    Print-Warning "HiFi-GAN repository already exists, skipping clone..."
}

# Download HiFi-GAN checkpoint
Print-Info "Checking HiFi-GAN checkpoint..."
if (-not (Test-Path "vocoder_checkpoints\LJ_FT_T2_V3")) {
    Print-Warning "HiFi-GAN checkpoint not found."
    Print-Info "Please download manually from:"
    Write-Host "https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y" -ForegroundColor Yellow
    Write-Host ""
    Print-Info "Steps:"
    Write-Host "1. Download the file LJ_FT_T2_V3.tar.gz"
    Write-Host "2. Extract it to vocoder_checkpoints\ directory"
    Write-Host "3. The final path should be: vocoder_checkpoints\LJ_FT_T2_V3\"
    Write-Host ""
    
    $response = Read-Host "Have you already downloaded and extracted it? (y/n)"
    if ($response -eq "y" -or $response -eq "Y") {
        if (Test-Path "vocoder_checkpoints\LJ_FT_T2_V3") {
            Print-Success "HiFi-GAN checkpoint found!"
        } else {
            Print-Warning "Checkpoint not found at expected location. Please verify."
        }
    }
} else {
    Print-Success "HiFi-GAN checkpoint already exists"
}

# Check if MFA aligned cache exists
Print-Info "Checking for MFA aligned cache..."
if (Test-Path "proper_cache_strict.pt") {
    Print-Success "MFA aligned cache (proper_cache_strict.pt) found!"
    Print-Info "You can skip dataset download and alignment."
} else {
    Print-Warning "MFA aligned cache not found."
    Write-Host ""
    Write-Host "You have two options:" -ForegroundColor Yellow
    Write-Host "  1. Place your pre-existing 'proper_cache_strict.pt' in the project root"
    Write-Host "  2. Download and align the dataset (see next step)"
    Write-Host ""
}

# Ask user if they want to download dataset
Write-Host ""
$response = Read-Host "Do you want to download the LJSpeech dataset? (y/n)"
if ($response -eq "y" -or $response -eq "Y") {
    Print-Info "Downloading and preparing LJSpeech dataset..."
    python download_datasets.py --dataset single-speaker
    Print-Success "Dataset downloaded and prepared"
    
    Write-Host ""
    Print-Warning "IMPORTANT: You need to run Montreal Forced Aligner (MFA)"
    Write-Host ""
    Write-Host "To install MFA (requires Anaconda/Miniconda):" -ForegroundColor Yellow
    Write-Host "  conda install -c conda-forge montreal-forced-aligner"
    Write-Host ""
    Write-Host "To align the dataset:" -ForegroundColor Yellow
    Write-Host "  mfa model download acoustic english_us_arpa"
    Write-Host "  mfa model download dictionary english_us_arpa"
    Write-Host "  mfa align data/training_data_ljspeech english_us_arpa english_us_arpa data/textgrid_data"
    Write-Host ""
    Print-Info "After alignment completes, the cache file will be generated on first training run."
} else {
    Print-Info "Skipping dataset download."
}

# Create test inference batch script
Print-Info "Creating test inference script..."
$testInferenceBat = @"
@echo off
REM Simple test inference script

echo Running test inference...

if not exist "checkpoints\run_stable\best.pt" (
    echo Error: No checkpoint found at checkpoints\run_stable\best.pt
    echo Please train the model first or specify a different checkpoint.
    exit /b 1
)

python spev_real_metrics.py ^
    --mode infer ^
    --checkpoint checkpoints\run_stable\best.pt ^
    --text "Hello world! This is a test of the SPEV text to speech system." ^
    --duration_scale 1.0 ^
    --pitch_scale 1.0 ^
    --output output.wav

echo Test complete! Check output.wav
pause
"@

Set-Content -Path "test_inference.bat" -Value $testInferenceBat
Print-Success "Test inference script created (test_inference.bat)"

# Create advanced (coordinator layer) inference batch script
# NOTE: --nasality/--valence/--arousal/--dominance/--age/--lung_capacity/
# --word_emphasis belonged to the old spev_advanced.py, which no longer
# exists. The current coordinator layer takes --emotion + inline [event]
# tags instead - see spev_embodied_core.py / spev_temporal_policy.py.
Print-Info "Creating advanced inference example script..."
$testAdvancedBat = @"
@echo off
REM Advanced inference: non-verbal events + time-varying emotion curves

echo Running advanced inference (coordinator layer)...

if not exist "checkpoints\run_stable\best.pt" (
    echo Error: No checkpoint found at checkpoints\run_stable\best.pt
    exit /b 1
)

echo.
echo 1. spev_embodied_core.py - inline non-verbal events ([sigh], [breath])
python spev_embodied_core.py ^
    --text "I am so tired... [sigh] but I must go on." ^
    --emotion exhausted ^
    --checkpoint checkpoints\run_stable\best.pt ^
    --hifigan_dir .\hifi-gan ^
    --output output_embodied.wav

echo.
echo 2. spev_temporal_policy.py - time-varying emotion curves
python spev_temporal_policy.py ^
    --text "Oh my god, I am so relieved." ^
    --emotion relief ^
    --checkpoint checkpoints\run_stable\best.pt ^
    --hifigan_dir .\hifi-gan ^
    --output output_temporal.wav

echo Advanced test complete! Check output_embodied.wav and output_temporal.wav
pause
"@

Set-Content -Path "test_advanced.bat" -Value $testAdvancedBat
Print-Success "Advanced inference script created (test_advanced.bat)"

# Summary
Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""
Print-Success "Environment setup finished successfully!"
Write-Host ""
Write-Host "📋 Next Steps:" -ForegroundColor Cyan
Write-Host ""

if (Test-Path "proper_cache_strict.pt") {
    Write-Host "1. " -NoNewline
    Print-Success "MFA cache found - Ready to train!"
    Write-Host "   Run: python spev_real_metrics.py --mode train --data_dir data/training_data_ljspeech --textgrid_dir data/textgrid_data --hifigan_dir vocoder_checkpoints/LJ_FT_T2_V3 --name run_stable --epochs 100"
} else {
    Write-Host "1. " -NoNewline
    Print-Warning "Prepare training data:"
    Write-Host "   a. Download dataset: python download_datasets.py --dataset single-speaker"
    Write-Host "   b. Install MFA (requires Anaconda): conda install -c conda-forge montreal-forced-aligner"
    Write-Host "   c. Download models: mfa model download acoustic english_us_arpa && mfa model download dictionary english_us_arpa"
    Write-Host "   d. Align data: mfa align data/training_data_ljspeech english_us_arpa english_us_arpa data/textgrid_data"
    Write-Host ""
    Write-Host "   OR"
    Write-Host ""
    Write-Host "   Place your pre-existing 'proper_cache_strict.pt' file in the project root"
}

Write-Host ""
Write-Host "2. Train the model:"
Write-Host "   python spev_real_metrics.py --mode train --data_dir data/training_data_ljspeech --textgrid_dir data/textgrid_data --hifigan_dir vocoder_checkpoints/LJ_FT_T2_V3 --name run_stable --epochs 100"
Write-Host ""
Write-Host "3. Test synthesis:"
Write-Host "   .\test_inference.bat"
Write-Host "   or"
Write-Host "   .\test_advanced.bat (for advanced voice controls)"
Write-Host ""
Write-Host "📚 Documentation:" -ForegroundColor Cyan
Write-Host "   - README.md - Complete documentation"
Write-Host "   - PRODUCTION_SYSTEM_SUMMARY.md - System architecture and deployment guide"
Write-Host ""
Print-Info "Remember to activate the virtual environment before running commands:"
Write-Host "   .\venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Happy synthesizing! 🎤✨" -ForegroundColor Magenta
Write-Host ""