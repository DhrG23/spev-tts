# SPEV TTS - Advanced Text-to-Speech System

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

**SPEV** (Speech Production with Emotional Voice) is a production-ready FastSpeech 2 based text-to-speech system with advanced voice control features including emotional intensity, voice quality modulation, and physiological constraints.

## ✨ Features

### Core TTS Capabilities
- **FastSpeech 2 Architecture** - Non-autoregressive, fast synthesis
- **Variance Predictors** - Duration, pitch, and energy modeling
- **MFA Alignment Support** - Strict phoneme-level alignment using Montreal Forced Aligner
- **HiFi-GAN Vocoder** - High-quality neural vocoding (with Griffin-Lim fallback)

### Advanced Voice Controls (spev_advanced.py)
- **Voice Quality Control**
  - Breathiness (0.0-1.0)
  - Roughness/Vocal Fry (0.0-1.0)
  - Nasality (0.0-1.0)

- **Emotional Intensity (VAD Model)**
  - Valence: Negative to Positive
  - Arousal: Calm to Excited
  - Dominance: Submissive to Dominant

- **Physiological Constraints**
  - Age-based pitch scaling (0-99 years)
  - Lung capacity for breath phrasing (0.3-1.0)

- **Word-Level Emphasis**
  - Per-word intensity control
  - Dynamic prosody modulation

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended, CPU supported)
- 8GB+ RAM
- Dependencies:
  ```
  torch>=2.0.0
  librosa>=0.10.0
  soundfile
  numpy
  textgrid
  cmudict
  ```

## 🚀 Quick Start

### Option 1: Using Quick Start Scripts

**Linux/macOS:**
```bash
chmod +x QUICKSTART.sh
./QUICKSTART.sh
```

**Windows:**
Use powershell (pwsh) not Windows Powershell (Default)
```powershell
.\QUICKSTART.ps1
```

### Option 2: Manual Setup

1. **Install Dependencies**
```bash
pip install torch torchaudio librosa soundfile numpy textgrid cmudict pandas requests
```

2. **Download HiFi-GAN Vocoder**
```bash
# Create vocoder directory
mkdir -p vocoder_checkpoints
cd vocoder_checkpoints

# Download LJ Speech fine-tuned HiFi-GAN
wget https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y?usp=sharing -O LJ_FT_T2_V3.tar.gz
tar -xzf LJ_FT_T2_V3.tar.gz
cd ..
```

3. **Clone HiFi-GAN Repository**
```bash
git clone https://github.com/jik876/hifi-gan.git
cd hifi-gan
pip install -r requirements.txt
cd ..
```

4. **Prepare Training Data**

**Option A: Download and Prepare Dataset**
```bash
# Download LJSpeech dataset
python download_datasets.py --dataset single-speaker

# This creates:
# - data/training_data_ljspeech/ (processed audio + transcripts)
```

**Option B: Align Your Own Data with MFA**
```bash
# Install Montreal Forced Aligner
conda install -c conda-forge montreal-forced-aligner

# Download acoustic model and dictionary
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

# Run alignment
mfa align data/training_data_ljspeech english_us_arpa english_us_arpa data/textgrid_data

# This creates TextGrid files with phoneme timings
```

## 🎓 Training

### Basic Training (Standard FastSpeech 2)
```bash
python spev_tts.py \
  --mode train \
  --data_dir data/training_data_ljspeech \
  --textgrid_dir data/textgrid_data \
  --hifigan_dir vocoder_checkpoints/LJ_FT_T2_V3 \
  --warmup_epochs 20 \
  --epochs 100
```

### Advanced Training (with Voice Controls)

Expressive data (cannot be skipped)

  If you want learned breathiness / emotion / vocal texture, you need:
    *expressive speech datasets
    *or manually labeled prosody
    *or multi-style speakers
  
  Without data, no architecture will fix this.

```bash
python spev_advanced.py \
  --mode train \
  --data_dir data/training_data_ljspeech \
  --textgrid_dir data/textgrid_data \
  --hifigan_dir vocoder_checkpoints/LJ_FT_T2_V3 \
  --warmup_epochs 20 \
  --epochs 150
```

**Training Notes:**
- **Warmup Phase**: First N epochs train duration predictor only
- **Variance Phase**: Remaining epochs train pitch/energy predictors
- **Checkpoints**: Saved every 10 epochs to `checkpoints/`
- **Cache**: First run creates `proper_cache_strict.pt` (subsequent runs load instantly)

## 🎤 Inference

### Basic Synthesis
```bash
python spev_tts.py \
  --mode infer \
  --checkpoint checkpoints/ckpt_100.pt \
  --text "Hello world! This is a test." \
  --duration_scale 1.0 \
  --pitch_scale 1.0
```

### Advanced Synthesis with Voice Controls
```bash
python spev_advanced.py \
  --mode infer \
  --checkpoint checkpoints/best_model.pt \
  --text "Hello world! This is amazing." \
  --breathiness 0.3 \
  --roughness 0.1 \
  --nasality 0.2 \
  --valence 0.8 \
  --arousal 0.6 \
  --dominance 0.5 \
  --age 35 \
  --lung_capacity 0.8 \
  --word_emphasis "1.0,1.0,2.5,1.0" \
  --output my_speech.wav
```

### Voice Control Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `--breathiness` | 0.0-1.0 | Adds breathy quality (higher = more air) |
| `--roughness` | 0.0-1.0 | Adds vocal fry/creak (higher = more rough) |
| `--nasality` | 0.0-1.0 | Nasal resonance (higher = more nasal) |
| `--valence` | 0.0-1.0 | Emotional valence (0=negative, 1=positive) |
| `--arousal` | 0.0-1.0 | Energy level (0=calm, 1=excited) |
| `--dominance` | 0.0-1.0 | Authority (0=submissive, 1=dominant) |
| `--age` | 0-99 | Speaker age (affects pitch) |
| `--lung_capacity` | 0.3-1.0 | Breath phrasing (lower = more pauses) |
| `--word_emphasis` | CSV floats | Per-word emphasis weights |

## 📊 Dataset Preparation

### Supported Datasets
1. **LJSpeech** (Single-speaker, ~24 hours)
2. **LibriTTS-R** (Multi-speaker)

### Download Script
```bash
# Download LJSpeech
python download_datasets.py --dataset single-speaker

# Download LibriTTS (multi-speaker)
python download_datasets.py --dataset multi-speaker

# Download both
python download_datasets.py --dataset both
```

### Custom Dataset Format
If using your own data:
```
data/
  my_dataset/
    audio1.wav
    audio1.txt
    audio2.wav
    audio2.txt
```

Requirements:
- 22050 Hz sample rate
- Mono audio
- Text files with same basename as audio

## 🏗️ Architecture Overview

### Model Components
1. **Phoneme Encoder** - Processes input phoneme sequence
2. **Variance Adaptors** - Predicts duration, pitch, energy
3. **Length Regulator** - Expands phoneme features to frame-level
4. **Mel Decoder** - Generates mel-spectrograms
5. **Vocoder** - Converts mels to waveforms (HiFi-GAN or Griffin-Lim)

### Advanced Components (spev_advanced.py)
- **VAD Emotion Embedding** - 3D spherical emotion space
- **Voice Quality Predictors** - Breathiness, roughness, nasality heads
- **Breath Generator** - Physiologically-constrained pause insertion
- **Age Embedding** - Pitch scaling based on vocal fold characteristics

## 🔧 Troubleshooting

### Common Issues

**1. "No TextGrid files found"**
- Ensure MFA alignment completed successfully
- Check `--textgrid_dir` path matches MFA output directory

**2. "HiFi-GAN not found, using Griffin-Lim"**
- Download HiFi-GAN checkpoint (see Quick Start)
- Verify `--hifigan_dir` contains `config.json` and `g_*` files

**3. "Word not in CMU dict"**
- The word isn't in CMU pronunciation dictionary
- Model falls back to `<SIL>` (silence) token
- Consider pre-processing text to use known words

**4. Cache file errors**
- Delete `proper_cache_strict.pt` and regenerate
- Ensure sufficient disk space

**5. CUDA out of memory**
- Reduce batch size in training code (default: 16)
- Use smaller model dimensions
- Use CPU: Model automatically falls back if no GPU

## 📈 Performance Tips

### Training
- **GPU**: 4-6 hours on RTX 3090 (100 epochs)
- **CPU**: ~24-48 hours (not recommended)
- **Batch Size**: Adjust based on GPU memory (16 works for 8GB VRAM)

### Inference
- **Real-time Factor**: ~0.05 on GPU (20x faster than real-time)
- **Cold Start**: ~2-3 seconds (model loading)
- **Warm Inference**: ~100ms per sentence

## 📚 References

- FastSpeech 2: [Paper](https://arxiv.org/abs/2006.04558)
- HiFi-GAN: [Paper](https://arxiv.org/abs/2010.05646) | [GitHub](https://github.com/jik876/hifi-gan)
- Montreal Forced Aligner: [Docs](https://montreal-forced-aligner.readthedocs.io/)

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{spev_tts,
  title = {SPEV: Advanced Text-to-Speech with Emotional Voice Control},
  year = {2026},
  url = {https://github.com/dhrg23/spev-tts}
}
```

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 💬 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: dhruv23gautam@gamil.com

---

**Note**: This is a research/educational project. For production use, consider additional optimizations and testing.