# SPEV TTS - Advanced Text-to-Speech System

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

**SPEV** (Speech Production with Emotional Voice) is a FastSpeech 2 based
text-to-speech system with controllable voice quality, emotion, and
non-verbal sound (sighs, breaths) layered on top.

## 🏗️ Architecture

The project is organized in layers, each script building on the one below it:

```
spev_real_metrics.py       <- the engine ("muscle"): FastSpeech 2 model,
                               training loop, HiFi-GAN/Griffin-Lim vocoder.
                               Run this directly for plain TTS.
        ↑ imported by
spev_embodied_core.py      <- coordinator ("spinal cord"): adds inline
                               non-verbal events ([sigh], [breath], [grunt])
                               synthesized via DSP and mixed with speech.
        ↑ superseded by
spev_temporal_policy.py    <- upgraded coordinator: turns static voice
                               controls into time-varying curves (e.g.
                               breathiness fading from 1.0 → 0.0 for
                               "relief"), plus a scaffold LSTM policy model
                               that is architecturally present but **not
                               trained** — it currently runs in heuristic
                               (hand-authored curve) mode only.
```

All three scripts require the same trained checkpoint (produced by
`spev_real_metrics.py --mode train`).

### Data preparation
- **`download_datasets.py`** — downloads and formats LJSpeech
  (single-speaker) and/or LibriTTS-R (multi-speaker) into the
  `wav + matching .txt transcript` layout the trainer expects.
- **`advanced__download_dataset.py`** — converts already-downloaded
  emotional speech corpora (ESD, Jenny) into that same layout, since they
  don't ship in it natively. (Its own module docstring calls itself
  `spev_data_prep.py` — that's a leftover from a rename; the file to run is
  `advanced__download_dataset.py`.) This is the dataset you want if you plan
  to train real (not heuristic) emotional/expressive control — see the
  warning under Training below.
- **`generate_clean_requirements.py`** — regenerates `requirements.txt` from
  a conda environment export. Not needed for normal use.

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended; CPU works but is slow)
- 8GB+ RAM
- The [espeak-ng](https://github.com/espeak-ng/espeak-ng) system package,
  required by the `phonemizer` library used for text → phoneme conversion:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install espeak-ng
  # macOS
  brew install espeak-ng
  ```

Install Python dependencies:
```bash
pip install -r requirements.txt
```
`requirements.txt` now includes `torch`/`torchaudio` and `phonemizer`
(these were missing before and are required by every script here).

## 🚀 Quick Start

### 1. Get training data
```bash
# LJSpeech (single-speaker, ~24 hours)
python download_datasets.py --dataset single-speaker

# LibriTTS-R (multi-speaker)
python download_datasets.py --dataset multi-speaker
```

### 2. (Optional) MFA alignment
```bash
conda install -c conda-forge montreal-forced-aligner
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa align data/training_data_ljspeech english_us_arpa english_us_arpa data/textgrid_data
```
If you skip this, `spev_real_metrics.py` falls back to uniform alignment
(it prints a warning if the `textgrid` package or TextGrid files aren't found).

### 3. Get a vocoder
```bash
mkdir -p vocoder_checkpoints && cd vocoder_checkpoints
wget https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y?usp=sharing -O LJ_FT_T2_V3.tar.gz
tar -xzf LJ_FT_T2_V3.tar.gz
cd ..
git clone https://github.com/jik876/hifi-gan.git
```
Without this, synthesis falls back to Griffin-Lim (noticeably lower quality).

## 🎓 Training

```bash
python spev_real_metrics.py \
  --mode train \
  --data_dir data/training_data_ljspeech \
  --textgrid_dir data/textgrid_data \
  --hifigan_dir vocoder_checkpoints/LJ_FT_T2_V3 \
  --epochs 100 \
  --batch_size 16 \
  --name run_stable
```

**Expressive data is not optional.** If you want breathiness/roughness/
emotion that's actually *learned* rather than hand-authored, you need
expressive/emotional speech data (e.g. ESD or Jenny, via
`advanced__download_dataset.py`) or manually labeled prosody. Training on
plain LJSpeech alone will not produce learned expressiveness — architecture
can't compensate for missing data. The `spev_embodied_core.py` /
`spev_temporal_policy.py` layers apply their voice-quality and emotion
effects as post-hoc DSP/heuristic controls regardless, so they'll work on a
plain LJSpeech checkpoint too — just know that in that case the "emotion"
comes entirely from the coordinator layer's rules, not from anything the
acoustic model learned.

Checkpoints save to `checkpoints/<run_name>/last.pt` and `best.pt` every
epoch; test inference runs automatically every 10 epochs.

## 🎤 Inference

### Plain synthesis (no coordinator layer)
```bash
python spev_real_metrics.py \
  --mode infer \
  --checkpoint checkpoints/run_stable/best.pt \
  --text "Hello world! This is a test." \
  --breathiness 0.1 --roughness 0.05 --brightness 0.0 \
  --pitch_scale 1.0 --duration_scale 1.0 --energy_scale 1.0 \
  --output output.wav
```

### With non-verbal events (sighs, breaths)
```bash
python spev_embodied_core.py \
  --text "I am so tired... [sigh] but I must go on." \
  --emotion exhausted \
  --checkpoint checkpoints/run_stable/best.pt \
  --hifigan_dir ./hifi-gan \
  --output embodied_output.wav
```
`--emotion` choices: `neutral`, `exhausted`, `excited`, `secretive`, `angry`.

### With time-varying (curve-based) emotion
```bash
python spev_temporal_policy.py \
  --text "Oh my god, I am so relieved." \
  --emotion relief \
  --checkpoint checkpoints/run_stable/best.pt \
  --hifigan_dir ./hifi-gan \
  --output temporal_output.wav
```
`--emotion` choices: `neutral`, `exhausted`, `relief`, `anxious`, `angry`
(a different set from `spev_embodied_core.py` — the two scripts weren't
kept in sync).

### Control parameters (`spev_real_metrics.py`)

| Parameter | Range | Description |
|-----------|-------|-------------|
| `--breathiness` | 0.0-0.8 | Breathy/airy voice quality |
| `--roughness` | 0.0-1.5 | Vocal fry / creak |
| `--brightness` | -2.5 to 2.5 | Spectral tilt |
| `--pitch_scale` | float | Pitch scaling factor |
| `--duration_scale` | float | Speaking-rate scaling factor |
| `--energy_scale` | float | Loudness scaling factor |

## 🏗️ Model Architecture

1. **Phoneme Encoder** — 4 FFT blocks (self-attention + conv feed-forward)
   over espeak/`phonemizer` IPA output. Note: tokenization is done as
   `list(phonemize(text))`, which splits at the *character* level, not at
   discrete multi-character phoneme units — a simplification, not a bug
   (it's applied consistently at train and inference time), but worth
   knowing if you extend the vocabulary logic.
2. **Variance Adaptors** — duration, pitch, energy, breathiness, roughness,
   and brightness predictors, each with output clamping to prevent
   exploding values during training.
3. **Length Regulator** — expands phoneme-level features to frame level.
4. **Mel Decoder** — 4 more FFT blocks, outputs 80-bin mel-spectrograms.
5. **Vocoder** — HiFi-GAN if `--hifigan_dir` has a valid checkpoint,
   otherwise Griffin-Lim.

## 🔧 Troubleshooting

**"No TextGrid files found"** — MFA alignment didn't complete, or
`--textgrid_dir` doesn't match MFA's output directory. Training will fall
back to uniform alignment either way.

**"HiFi-GAN not found, using Griffin-Lim"** — download the HiFi-GAN
checkpoint (Quick Start step 3) and confirm `--hifigan_dir` points to a
folder with `config.json` and a `g_*` generator file.

**`ModuleNotFoundError: espeak` / phonemizer backend errors** — install the
`espeak-ng` system package (see Requirements); `pip install phonemizer`
alone isn't enough.

**Cache/data errors** — delete `proper_cache_strict.pt` if present and
re-run; ensure sufficient disk space.

**CUDA out of memory** — lower `--batch_size` (default 16), or run on CPU
(automatic fallback if no GPU is detected).

**Console scripts after `pip install -e .`** — `setup.py`'s entry points
now match the current files: `spev-run` (train/infer via
`spev_real_metrics.py`'s own `--mode` flag), `spev-embodied-infer`,
`spev-temporal-infer`, `spev-download`, and `spev-prep-dataset`. Running
the scripts directly with `python spev_real_metrics.py ...` etc. still
works identically and needs no install step.

## 📈 Performance Notes

These are the original project's estimates and haven't been re-verified
against the current codebase:
- Training: ~4-6 hours on an RTX 3090 for 100 epochs; CPU is 24-48+ hours.
- Inference: roughly real-time-factor 0.05 on GPU (~20x faster than
  real-time) once warm; a few seconds of cold-start for model loading.

## 📚 References

- FastSpeech 2: [Paper](https://arxiv.org/abs/2006.04558)
- HiFi-GAN: [Paper](https://arxiv.org/abs/2010.05646) | [GitHub](https://github.com/jik876/hifi-gan)
- Montreal Forced Aligner: [Docs](https://montreal-forced-aligner.readthedocs.io/)
- phonemizer: [GitHub](https://github.com/bootphon/phonemizer)

## 📄 License

MIT License - see LICENSE file for details.

---

**Note**: This is a research/educational project. For production use,
consider additional testing, the `setup.py` entry-point fix mentioned
above, and re-validating the performance numbers on current hardware.
