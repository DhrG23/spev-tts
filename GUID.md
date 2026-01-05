# SPEV TTS - Installation Guide

This guide covers multiple installation methods for SPEV TTS.

## 📦 Installation Methods

### Method 1: Editable Install (Recommended for Development)

This method allows you to modify the code and see changes immediately without reinstalling.

```bash
# Clone the repository
git clone https://github.com/dhrg23/spev-tts.git
cd spev-tts

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .
```

### Method 2: Standard Install from Source

```bash
# Clone the repository
git clone https://github.com/dhrg23/spev-tts.git
cd spev-tts

# Install
pip install .
```

### Method 3: Install with Optional Dependencies

```bash
# Install with development tools
pip install -e ".[dev]"

# Install with training utilities (tensorboard, wandb)
pip install -e ".[training]"

# Install with documentation tools
pip install -e ".[docs]"

# Install with alignment tools
pip install -e ".[alignment]"

# Install everything
pip install -e ".[all]"
```

### Method 4: Quick Start Script (Full Setup)

The quickest way to get everything set up:

```bash
# Linux/macOS
chmod +x QUICKSTART.sh
./QUICKSTART.sh

# Windows
.\QUICKSTART.ps1
```

This script will:
- Create virtual environment
- Install all dependencies
- Download HiFi-GAN vocoder
- Set up directory structure
- Create helper scripts
- Optionally download dataset and run MFA alignment

## 🔧 Manual Installation Steps

If you prefer to install dependencies manually:

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install PyTorch

**For CUDA (GPU):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**
```bash
pip install torch torchaudio
```

### 3. Install Core Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install librosa soundfile numpy scipy textgrid cmudict pandas requests tqdm
```

### 4. Install SPEV Package

```bash
pip install -e .
```

## 🎯 Verifying Installation

After installation, verify everything works:

```bash
# Check Python packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import librosa; print(f'Librosa: {librosa.__version__}')"
python -c "import soundfile; print(f'SoundFile: {soundfile.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check command-line tools (if installed with setup.py)
spev-train --help
spev-infer --help
```

Or use the system check script:
```bash
./check_system.sh  # Linux/macOS
```

## 📋 System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 8GB RAM
- 10GB disk space
- CPU with AVX support

### Recommended Requirements
- Python 3.9+
- 16GB RAM
- 20GB disk space
- NVIDIA GPU with 6GB+ VRAM
- CUDA 11.8+

## 🔍 Installation Options Explained

### Core Installation
```bash
pip install -e .
```
Includes only essential dependencies for running inference.

### Development Installation
```bash
pip install -e ".[dev]"
```
Adds:
- pytest (testing framework)
- black (code formatter)
- flake8 (linter)
- mypy (type checker)
- isort (import sorter)

### Training Installation
```bash
pip install -e ".[training]"
```
Adds:
- tensorboard (experiment tracking)
- wandb (cloud experiment tracking)
- matplotlib (visualization)
- seaborn (statistical plots)

### Complete Installation
```bash
pip install -e ".[all]"
```
Includes all optional dependencies.

## 🐛 Troubleshooting

### Issue: "command 'gcc' failed"

**Solution:** Install build tools
```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# macOS
xcode-select --install

# Windows
# Install Visual Studio Build Tools from Microsoft
```

### Issue: "Could not find a version that satisfies the requirement torch"

**Solution:** Upgrade pip and setuptools
```bash
pip install --upgrade pip setuptools wheel
```

### Issue: "No module named 'soundfile'"

**Solution:** Install libsndfile system library
```bash
# Ubuntu/Debian
sudo apt-get install libsndfile1

# macOS
brew install libsndfile

# Windows
# Usually works without additional libraries, but if needed:
# conda install -c conda-forge libsndfile
```

### Issue: "ModuleNotFoundError: No module named 'textgrid'"

**Solution:**
```bash
pip install textgrid
```

### Issue: "CUDA out of memory" during training

**Solutions:**
1. Reduce batch size in training code (edit line: `batch_size=16` → `batch_size=8`)
2. Use gradient accumulation
3. Use CPU training (slower but no memory limit)

### Issue: "HiFi-GAN not found, using Griffin-Lim"

**Solution:** Download HiFi-GAN checkpoint
```bash
cd vocoder_checkpoints
wget https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y?usp=sharing -O LJ_FT_T2_V3.tar.gz
tar -xzf LJ_FT_T2_V3.tar.gz
cd ..
```

## 🔄 Updating SPEV

If you installed in editable mode (`-e`), simply pull the latest changes:

```bash
git pull origin main
```

If you need to update dependencies:
```bash
pip install -e . --upgrade
```

## 🗑️ Uninstalling

To uninstall SPEV:

```bash
pip uninstall spev-tts
```

To remove everything including virtual environment:
```bash
deactivate  # Exit virtual environment
rm -rf venv  # Remove virtual environment directory
```

## 📚 Next Steps

After installation:

1. **Download Training Data**
   ```bash
   python download_datasets.py --dataset single-speaker
   ```

2. **Set Up MFA Alignment**
   ```bash
   conda install -c conda-forge montreal-forced-aligner
   mfa model download acoustic english_us_arpa
   mfa model download dictionary english_us_arpa
   mfa align data/training_data_ljspeech english_us_arpa english_us_arpa data/textgrid_data
   ```

3. **Train Model**
   ```bash
   ./train_model.sh
   # or manually:
   python spev_tts.py --mode train --data_dir data/training_data_ljspeech \
       --textgrid_dir data/textgrid_data --hifigan_dir vocoder_checkpoints/LJ_FT_T2_V3
   ```

4. **Test Synthesis**
   ```bash
   ./test_inference.sh
   # or manually:
   python spev_tts.py --mode infer --checkpoint checkpoints/best_model.pt \
       --text "Hello world!"
   ```

## 🆘 Getting Help

- **Documentation**: See README.md
- **Issues**: GitHub Issues

---

**Installation complete!** Proceed to training or inference as needed.