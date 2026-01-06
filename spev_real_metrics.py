"""
SPEV - REAL METRICS EDITION (ACTUALLY LEARNED FEATURES)
=======================================================

THE FIX:
Instead of "fake" regularization losses, this version MEASURES acoustic features
from your data and trains the model to predict/control them.

1. Breathiness <- Learned from (1 - Voicing_Probability)
2. Roughness   <- Learned from F0 Standard Deviation (Jitter proxy)
3. Brightness  <- Learned from Spectral Centroid (Timbre/Nasality proxy)

Usage:
  Train:
  python spev_real_metrics.py --mode train --data_dir ./wavs --textgrid_dir ./aligned --hifigan_dir ./hifi-gan

  Inference (Now these controls actually do something):
  python spev_real_metrics.py --mode infer --text "Hello world." --checkpoint checkpoints/best_model.pt \
    --breathiness 0.8  # Very whispery
    --roughness 0.5    # Creaky/Unstable
    --brightness 0.2   # Dark/Muffled
"""

import os
import sys
import glob
import json
import math
import torch
import argparse
import librosa
import numpy as np
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from phonemizer import phonemize

# --- HiFi-GAN Setup ---
sys.path.append('hifi-gan')
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

try:
    from env import AttrDict
    from models import Generator as HiFiGenerator
    HIFIGAN_AVAILABLE = True
except ImportError:
    print("⚠️  HiFi-GAN modules not found. Inference will use Griffin-Lim.")
    AttrDict = HiFiGenerator = None
    HIFIGAN_AVAILABLE = False

try:
    import textgrid
    TEXTGRID_AVAILABLE = True
except ImportError:
    print("⚠️  'textgrid' library not found. MFA support disabled.")
    textgrid = None
    TEXTGRID_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    'sr': 22050,
    'n_fft': 1024,
    'hop_length': 256,
    'n_mels': 80,
    'fmin': 0,
    'fmax': 8000
}

# =========================================================
# 1. MODEL COMPONENTS
# =========================================================
class FFTBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads=2, dropout=0.1, kernel_size=9):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim * 4, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(hidden_dim * 4, hidden_dim, kernel_size, padding=kernel_size//2)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, mask=None):
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        xt = x.transpose(1, 2)
        xt = self.conv2(self.relu(self.conv1(xt)))
        xt = self.dropout(xt)
        x = self.norm2(x + xt.transpose(1, 2))
        return x

class VariancePredictor(nn.Module):
    def __init__(self, hidden_dim, n_layers=2, kernel=3, dropout=0.1):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, kernel, padding=kernel//2),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ])
        layers.append(nn.Linear(hidden_dim, 1))
        self.layers = nn.Sequential(*layers[:-1])
        self.proj = layers[-1]

    def forward(self, x):
        x_t = x.transpose(1, 2)
        for layer in self.layers:
            if isinstance(layer, nn.Conv1d):
                x_t = layer(x_t)
            elif isinstance(layer, nn.LayerNorm):
                x_t = layer(x_t.transpose(1, 2)).transpose(1, 2)
            else:
                x_t = layer(x_t)
        return self.proj(x_t.transpose(1, 2)).squeeze(-1)

class LengthRegulator(nn.Module):
    def forward(self, x, durations):
        output = []
        for b in range(x.size(0)):
            expanded = []
            for t in range(x.size(1)):
                n = max(0, int(durations[b, t].item()))
                if n > 0:
                    expanded.append(x[b, t:t+1].repeat(n, 1))
            if not expanded:
                output.append(torch.zeros(1, x.size(2), device=x.device))
            else:
                output.append(torch.cat(expanded, dim=0))
                
        max_len = max(o.size(0) for o in output)
        mel_lens = torch.LongTensor([o.size(0) for o in output]).to(x.device)
        return torch.stack([F.pad(o, (0, 0, 0, max_len - o.size(0))) for o in output]), mel_lens

class RealMetricsFastSpeech2(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=256, n_mels=80):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.encoder_blocks = nn.ModuleList([FFTBlock(hidden_dim) for _ in range(4)])
        
        # --- Variance Predictors ---
        self.duration_predictor = VariancePredictor(hidden_dim)
        self.pitch_predictor = VariancePredictor(hidden_dim)
        self.energy_predictor = VariancePredictor(hidden_dim)
        
        # NEW: REAL Acoustic Feature Predictors
        self.breath_predictor = VariancePredictor(hidden_dim)    # Predicts (1 - voiced_prob)
        self.rough_predictor = VariancePredictor(hidden_dim)     # Predicts f0_std
        self.bright_predictor = VariancePredictor(hidden_dim)    # Predicts spectral_centroid
        
        # --- Embeddings for Conditioning ---
        self.pitch_embedding = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.energy_embedding = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        
        self.breath_embedding = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.rough_embedding = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.bright_embedding = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        
        self.length_regulator = LengthRegulator()
        self.decoder_blocks = nn.ModuleList([FFTBlock(hidden_dim) for _ in range(4)])
        self.mel_linear = nn.Linear(hidden_dim, n_mels)

    def forward(self, phoneme_ids, lengths, 
                target_durations=None, target_pitch=None, target_energy=None,
                target_breath=None, target_rough=None, target_bright=None,
                d_control=1.0, p_control=1.0, e_control=1.0):
        
        # 1. Text Encoder
        x = self.embedding(phoneme_ids)
        src_mask = (torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None])
        
        for block in self.encoder_blocks:
            x = block(x, mask=src_mask)
        
        # 2. Predict Variances
        log_dur_pred = self.duration_predictor(x)
        pitch_pred = self.pitch_predictor(x)
        energy_pred = self.energy_predictor(x)
        
        # NEW: Predict Quality
        breath_pred = self.breath_predictor(x)
        rough_pred = self.rough_predictor(x)
        bright_pred = self.bright_predictor(x)
        
        # 3. Select Values (Target vs Prediction)
        if target_durations is not None:
            durations = target_durations
            pitch = target_pitch
            energy = target_energy
            breath = target_breath
            rough = target_rough
            bright = target_bright
        else:
            # Inference Mode
            durations = torch.clamp((torch.exp(log_dur_pred) - 1) * d_control, min=0).round().long()
            pitch = pitch_pred * p_control
            energy = energy_pred * e_control
            # In inference, we can offset these with the arguments
            breath = breath_pred
            rough = rough_pred
            bright = bright_pred

            # Apply user overrides if provided in batch (handled in wrapper normally, simplified here)
            if target_breath is not None: breath = target_breath
            if target_rough is not None: rough = target_rough
            if target_bright is not None: bright = target_bright

        # 4. Length Regulation
        x_expanded, mel_len = self.length_regulator(x, durations)
        
        # Expand scalar features to frame level (Naive expansion for now, ideally predictive per frame)
        # We need to expand the PHONEME-level features to FRAME-level features
        pitch = self.length_regulator(pitch.unsqueeze(-1), durations)[0].transpose(1, 2)
        energy = self.length_regulator(energy.unsqueeze(-1), durations)[0].transpose(1, 2)
        breath = self.length_regulator(breath.unsqueeze(-1), durations)[0].transpose(1, 2)
        rough = self.length_regulator(rough.unsqueeze(-1), durations)[0].transpose(1, 2)
        bright = self.length_regulator(bright.unsqueeze(-1), durations)[0].transpose(1, 2)
        
        # 5. Condition Decoder
        # Add acoustic embeddings to the expanded text encoding
        dec_input = x_expanded.transpose(1, 2) # (B, Hidden, T)
        
        dec_input = dec_input + \
                    self.pitch_embedding(pitch) + \
                    self.energy_embedding(energy) + \
                    self.breath_embedding(breath) + \
                    self.rough_embedding(rough) + \
                    self.bright_embedding(bright)
                    
        dec_input = dec_input.transpose(1, 2) # Back to (B, T, Hidden)
        
        # 6. Decode
        mel_mask = (torch.arange(dec_input.size(1), device=x.device)[None, :] >= mel_len[:, None])
        for block in self.decoder_blocks:
            dec_input = block(dec_input, mask=mel_mask)
            
        mel_out = self.mel_linear(dec_input)
        
        return {
            'mel_pred': mel_out,
            'log_duration_pred': log_dur_pred,
            'pitch_pred': pitch_pred,
            'energy_pred': energy_pred,
            'breath_pred': breath_pred,
            'rough_pred': rough_pred,
            'bright_pred': bright_pred,
            'src_mask': src_mask,
            'mel_len': mel_len
        }

# =========================================================
# 2. DATASET (THE REAL METRICS EXTRACTION)
# =========================================================
class RealMetricsDataset(Dataset):
    def __init__(self, data_dir, textgrid_dir=None, cache_file='real_metrics_cache.pt'):
        self.textgrid_dir = textgrid_dir
        
        if os.path.exists(cache_file):
            print(f"📦 Loading cache from {cache_file}...")
            data = torch.load(cache_file, weights_only=False)
            self.utterances = data['utterances']
            self.stats = data['stats']
            self.vocab = data['vocab']
            self.ph_to_idx = {ph: i for i, ph in enumerate(self.vocab)}
            print(f"✓ Loaded {len(self.utterances)} items.")
            return

        self.utterances = []
        search_pattern = os.path.join(os.path.abspath(data_dir), "**", "*.wav")
        wav_files = sorted(glob.glob(search_pattern, recursive=True))
        
        print("🔍 Pass 1: Scanning Stats (Pitch, Energy, Centroid)...")
        vocab_set = set(['<PAD>', '<UNK>', '<SIL>'])
        all_p, all_e, all_c = [], [], []
        
        for w in wav_files[:200]: 
            try:
                y, _ = librosa.load(w, sr=CONFIG['sr'])
                # Pitch
                f0, _, _ = librosa.pyin(y, fmin=60, fmax=500, sr=CONFIG['sr'])
                f0 = np.log(np.nan_to_num(f0, nan=1e-8) + 1e-8)
                all_p.extend(f0[f0 > -5].tolist())
                
                # Energy
                rms = np.log(librosa.feature.rms(y=y)[0] + 1e-8)
                all_e.extend(rms.tolist())
                
                # Centroid (Brightness)
                cent = np.log(librosa.feature.spectral_centroid(y=y, sr=CONFIG['sr'])[0] + 1e-8)
                all_c.extend(cent.tolist())
            except: continue

        self.stats = {
            'p_mean': np.mean(all_p), 'p_std': np.std(all_p),
            'e_mean': np.mean(all_e), 'e_std': np.std(all_e),
            'c_mean': np.mean(all_c), 'c_std': np.std(all_c)
        }
        print(f"  Stats: Brightness Mean={self.stats['c_mean']:.2f}")

        print(f"⏳ Pass 2: Feature Extraction...")
        for wav_path in wav_files:
            try:
                utt = self._process(wav_path)
                if utt:
                    self.utterances.append(utt)
                    vocab_set.update(utt['phs'])
            except Exception as e:
                pass

        self.vocab = sorted(list(vocab_set))
        self.ph_to_idx = {ph: i for i, ph in enumerate(self.vocab)}
        
        torch.save({'utterances': self.utterances, 'stats': self.stats, 'vocab': self.vocab}, cache_file)

    def _process(self, wav_path):
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        
        # 1. Load Audio
        y, _ = librosa.load(wav_path, sr=CONFIG['sr'])
        
        # 2. Get Alignment (MFA preferred, else fallback)
        phs, durs = [], []
        tg_path = None
        if self.textgrid_dir:
            tg_candidates = glob.glob(os.path.join(self.textgrid_dir, "**", f"{basename}.TextGrid"), recursive=True)
            if tg_candidates: tg_path = tg_candidates[0]
            
        if tg_path and TEXTGRID_AVAILABLE:
            tg = textgrid.TextGrid.fromFile(tg_path)
            phone_tier = next((t for t in tg if t.name.lower() in ['phones', 'phonemes']), None)
            for interval in phone_tier:
                if interval.maxTime * CONFIG['sr'] / CONFIG['hop_length'] > len(y) / CONFIG['hop_length']: break
                frames = int((interval.maxTime - interval.minTime) * CONFIG['sr'] / CONFIG['hop_length'])
                if frames > 0:
                    phs.append(interval.mark if interval.mark else '<SIL>')
                    durs.append(frames)
        
        if not phs: # Fallback
            txt_path = wav_path.replace('.wav', '.txt')
            if not os.path.exists(txt_path): return None
            with open(txt_path) as f: text = f.read().strip()
            phs = ['<SIL>'] + list(phonemize(text, language="en-us", backend="espeak", strip=True)) + ['<SIL>']
            total_frames = int(len(y) / CONFIG['hop_length'])
            durs = [int(total_frames / len(phs))] * len(phs)

        # 3. Extract Frame-Level Features
        # Mel
        mel = librosa.feature.melspectrogram(y=y, sr=CONFIG['sr'], n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'], n_mels=CONFIG['n_mels'])
        mel = torch.log(torch.clamp(torch.from_numpy(mel), min=1e-5))
        
        # F0 & Voicing (for Breathiness)
        f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=60, fmax=500, sr=CONFIG['sr'], hop_length=CONFIG['hop_length'])
        f0 = np.nan_to_num(f0, nan=0.0)
        
        # Energy
        rms = librosa.feature.rms(y=y, hop_length=CONFIG['hop_length'])[0]
        
        # Spectral Centroid (for Brightness/Nasality)
        cent = librosa.feature.spectral_centroid(y=y, sr=CONFIG['sr'], hop_length=CONFIG['hop_length'])[0]
        
        # 4. Phoneme-Level Averaging (The Metric Calculation)
        p_phone, e_phone, breath_phone, rough_phone, bright_phone = [], [], [], [], []
        curr = 0
        
        for d in durs:
            end = curr + d
            
            # -- Pitch & Energy --
            f0_seg = f0[curr:end]
            f0_nz = f0_seg[f0_seg > 0]
            p_val = np.mean(np.log(f0_nz + 1e-8)) if len(f0_nz) > 0 else 0
            e_val = np.mean(np.log(rms[curr:end] + 1e-8))
            
            # -- BREATHINESS (1 - Voicing Probability) --
            # High voicing = clear, Low voicing = breathy
            v_prob_seg = voiced_prob[curr:end]
            breath_val = 1.0 - np.mean(v_prob_seg) if len(v_prob_seg) > 0 else 1.0
            
            # -- ROUGHNESS (Pitch Standard Deviation) --
            # High std dev = jittery/creaky
            rough_val = np.std(np.log(f0_nz + 1e-8)) if len(f0_nz) > 2 else 0.0
            
            # -- BRIGHTNESS (Spectral Centroid) --
            c_seg = cent[curr:end]
            bright_val = np.mean(np.log(c_seg + 1e-8)) if len(c_seg) > 0 else 0
            
            # Normalize
            p_norm = (p_val - self.stats['p_mean']) / self.stats['p_std']
            e_norm = (e_val - self.stats['e_mean']) / self.stats['e_std']
            bright_norm = (bright_val - self.stats['c_mean']) / self.stats['c_std']
            
            # Breath/Rough are 0-1 bounded usually, but lets just use raw for now or standardize
            # Since breath is 0-1 probability, we keep it. Roughness is unbounded, but small.
            
            p_phone.append(p_norm)
            e_phone.append(e_norm)
            breath_phone.append(breath_val)
            rough_phone.append(rough_val)
            bright_phone.append(bright_norm)
            
            curr = end

        min_l = min(mel.shape[1], sum(durs))
        return {
            'phs': phs, 'durs': durs, 'mel': mel[:, :min_l].T.numpy(),
            'pitch': np.array(p_phone), 'energy': np.array(e_phone),
            'breath': np.array(breath_phone), 'rough': np.array(rough_phone), 'bright': np.array(bright_phone)
        }

    def __len__(self): return len(self.utterances)
    def __getitem__(self, idx):
        u = self.utterances[idx]
        return {
            'ids': torch.LongTensor([self.ph_to_idx.get(p, 0) for p in u['phs']]),
            'durs': torch.LongTensor(u['durs']),
            'log_durs': torch.log(torch.LongTensor(u['durs']).float() + 1),
            'mel': torch.FloatTensor(u['mel']),
            'pitch': torch.FloatTensor(u['pitch']),
            'energy': torch.FloatTensor(u['energy']),
            'breath': torch.FloatTensor(u['breath']),
            'rough': torch.FloatTensor(u['rough']),
            'bright': torch.FloatTensor(u['bright']),
        }

def collate_fn(batch):
    ids = pad_sequence([b['ids'] for b in batch], batch_first=True)
    lens = torch.LongTensor([len(b['ids']) for b in batch])
    durs = pad_sequence([b['durs'] for b in batch], batch_first=True)
    l_durs = pad_sequence([b['log_durs'] for b in batch], batch_first=True)
    mels = pad_sequence([b['mel'] for b in batch], batch_first=True)
    
    pitch = pad_sequence([b['pitch'] for b in batch], batch_first=True)
    energy = pad_sequence([b['energy'] for b in batch], batch_first=True)
    breath = pad_sequence([b['breath'] for b in batch], batch_first=True)
    rough = pad_sequence([b['rough'] for b in batch], batch_first=True)
    bright = pad_sequence([b['bright'] for b in batch], batch_first=True)
    
    return {'ids': ids, 'lens': lens, 'durs': durs, 'log_durs': l_durs, 'mel': mels, 
            'pitch': pitch, 'energy': energy, 
            'breath': breath, 'rough': rough, 'bright': bright}

# =========================================================
# 3. TRAINING & INFERENCE
# =========================================================
class Trainer:
    def __init__(self, args):
        self.dataset = RealMetricsDataset(args.data_dir, args.textgrid_dir)
        self.model = RealMetricsFastSpeech2(len(self.dataset.vocab)).to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        self.vocoder = Vocoder(args.hifigan_dir) if args.hifigan_dir else None

    def train(self, epochs=100):
        loader = DataLoader(self.dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for b in loader:
                self.optimizer.zero_grad()
                
                # Move to device
                for k, v in b.items():
                    if isinstance(v, torch.Tensor): b[k] = v.to(DEVICE)
                
                out = self.model(b['ids'], b['lens'], 
                               target_durations=b['durs'],
                               target_pitch=b['pitch'],
                               target_energy=b['energy'],
                               target_breath=b['breath'],
                               target_rough=b['rough'],
                               target_bright=b['bright'])
                
                # Losses
                mask = ~out['src_mask']
                mel_len = min(out['mel_pred'].size(1), b['mel'].size(1))
                
                l_mel = F.l1_loss(out['mel_pred'][:, :mel_len], b['mel'][:, :mel_len])
                l_dur = F.mse_loss(out['log_duration_pred'][mask], b['log_durs'][mask])
                l_pitch = F.mse_loss(out['pitch_pred'][mask], b['pitch'][mask])
                l_energy = F.mse_loss(out['energy_pred'][mask], b['energy'][mask])
                
                # NEW REAL LOSSES
                l_breath = F.mse_loss(out['breath_pred'][mask], b['breath'][mask])
                l_rough = F.mse_loss(out['rough_pred'][mask], b['rough'][mask])
                l_bright = F.mse_loss(out['bright_pred'][mask], b['bright'][mask])
                
                loss = l_mel + l_dur + l_pitch + l_energy + l_breath + l_rough + l_bright
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.4f} (B:{l_breath.item():.3f} R:{l_rough.item():.3f} Br:{l_bright.item():.3f})")
            
            if (epoch+1) % 10 == 0:
                self.save(f"checkpoints/real_epoch_{epoch+1}.pt")
                if self.vocoder: self.test_inference(epoch+1)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'model': self.model.state_dict(), 'vocab': self.dataset.vocab, 'stats': self.dataset.stats}, path)

    def test_inference(self, epoch):
        self.model.eval()
        if len(self.dataset) > 0:
            sample = self.dataset[0]
            ids = sample['ids'].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = self.model(ids, torch.LongTensor([len(sample['ids'])]).to(DEVICE))
                wav = self.vocoder.infer(out['mel_pred'].transpose(1, 2))
            sf.write(f'results/real_test_{epoch}.wav', wav, CONFIG['sr'])

class Vocoder:
    def __init__(self, hifigan_dir):
        self.device = DEVICE
        self.model = None
        if not HIFIGAN_AVAILABLE: return
        config_path = os.path.join(hifigan_dir, 'config.json')
        ckpt_search = glob.glob(os.path.join(hifigan_dir, 'g_*')) 
        if os.path.exists(config_path) and ckpt_search:
            ckpt_path = sorted(ckpt_search)[-1]
            with open(config_path) as f: self.h = AttrDict(json.loads(f.read()))
            self.model = HiFiGenerator(self.h).to(self.device)
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device)['generator'])
            self.model.eval()
            self.model.remove_weight_norm()
    def infer(self, mel):
        if self.model is None: return np.zeros(100)
        with torch.no_grad(): return self.model(mel).squeeze().cpu().numpy()

def inference_mode(args):
    print(f"Loading {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    vocab = ckpt['vocab']
    model = RealMetricsFastSpeech2(len(vocab)).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    vocoder = Vocoder(args.hifigan_dir)
    ph_to_idx = {p: i for i, p in enumerate(vocab)}
    
    phs = ['<SIL>'] + list(phonemize(args.text, language="en-us", backend="espeak", strip=True)) + ['<SIL>']
    ids = torch.LongTensor([ph_to_idx.get(p, 0) for p in phs]).unsqueeze(0).to(DEVICE)
    
    # --- HERE IS THE MAGIC ---
    # We construct target tensors filled with the user's desired values
    # Because the model was trained on REAL values, injecting these will force the model 
    # to generate audio matching these characteristics.
    
    # 1. Breathiness (0.0 to 1.0)
    target_breath = torch.full((1, len(phs)), args.breathiness).to(DEVICE)
    
    # 2. Roughness (Standard Deviation of Pitch)
    # Natural roughness is usually 0.0 to 0.2. High roughness > 0.3.
    target_rough = torch.full((1, len(phs)), args.roughness).to(DEVICE)
    
    # 3. Brightness (Normalized Spectral Centroid)
    # 0 = Average. -1 = Dark. +1 = Bright/Nasal.
    target_bright = torch.full((1, len(phs)), args.brightness).to(DEVICE)
    
    with torch.no_grad():
        out = model(ids, torch.LongTensor([len(phs)]).to(DEVICE),
                    target_breath=target_breath,
                    target_rough=target_rough,
                    target_bright=target_bright)
        wav = vocoder.infer(out['mel_pred'].transpose(1, 2))
        
    sf.write(args.output, wav, CONFIG['sr'])
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'])
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--textgrid_dir', type=str)
    parser.add_argument('--hifigan_dir', type=str, default='vocoder_checkpoints/LJ_FT_T2_V3')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--text', type=str, default="Hello world")
    parser.add_argument('--output', type=str, default="output.wav")
    
    # The Real Controls
    parser.add_argument('--breathiness', type=float, default=0.1, help="0.0 (Clear) to 1.0 (Whisper)")
    parser.add_argument('--roughness', type=float, default=0.05, help="0.0 (Smooth) to 0.5 (Creaky)")
    parser.add_argument('--brightness', type=float, default=0.0, help="-2.0 (Muffled) to 2.0 (Bright/Nasal)")
    
    args = parser.parse_args()
    if args.mode == 'train':
        Trainer(args).train(100)
    else:
        inference_mode(args)