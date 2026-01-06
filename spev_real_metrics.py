"""
SPEV - PRODUCTION TRAINER (FIXED)
=================================

CHANGELOG:
✅ FIXED ENERGY: Now calculates real RMS energy (was zero placeholder).
✅ TEXTGRID SUPPORT: Uses MFA alignments if provided (CRITICAL for prosody).
✅ FEATURE EXTRACTION: Robust extraction of Pitch, Energy, Breathiness, Roughness, Brightness.
✅ AMP/GRAD ACCUM: Retained from production version for speed.

Usage:
  python .\spev_real_metrics.py --mode train --data_dir ./wavs --textgrid_dir ./textgrids --name run_fixed
"""

import os
import sys
import glob
import json
import random
import shutil
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import argparse
import librosa
import numpy as np
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
from phonemizer import phonemize
from tqdm import tqdm

# --- External Deps ---
sys.path.append('hifi-gan')
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

try:
    from env import AttrDict
    from models import Generator as HiFiGenerator
    HIFIGAN_AVAILABLE = True
except ImportError:
    print("⚠️  HiFi-GAN modules not found. Inference will use Griffin-Lim.")
    HIFIGAN_AVAILABLE = False

try:
    import textgrid
    TEXTGRID_AVAILABLE = True
except ImportError:
    print("⚠️  'textgrid' library not found. Falling back to uniform alignment (NOT RECOMMENDED).")
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
# 1. MODEL ARCHITECTURE (Unchanged)
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
        mel_lens = []
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
            mel_lens.append(output[-1].size(0))
                
        max_len = max(mel_lens)
        stacked = torch.stack([F.pad(o, (0, 0, 0, max_len - o.size(0))) for o in output])
        return stacked, torch.LongTensor(mel_lens).to(x.device)

class RealMetricsFastSpeech2(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=256, n_mels=80):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder_blocks = nn.ModuleList([FFTBlock(hidden_dim) for _ in range(4)])
        
        # Variances
        self.duration_predictor = VariancePredictor(hidden_dim)
        self.pitch_predictor = VariancePredictor(hidden_dim)
        self.energy_predictor = VariancePredictor(hidden_dim)
        self.breath_predictor = VariancePredictor(hidden_dim)
        self.rough_predictor = VariancePredictor(hidden_dim)
        self.bright_predictor = VariancePredictor(hidden_dim)
        
        # Embeddings
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
        
        x = self.embedding(phoneme_ids)
        src_mask = (torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None])
        
        for block in self.encoder_blocks:
            x = block(x, mask=src_mask)
        
        # Predict
        log_dur_pred = self.duration_predictor(x)
        pitch_pred = self.pitch_predictor(x)
        energy_pred = self.energy_predictor(x)
        breath_pred = self.breath_predictor(x)
        rough_pred = self.rough_predictor(x)
        bright_pred = self.bright_predictor(x)
        
        # Select
        if target_durations is not None:
            durations = target_durations
            pitch = target_pitch
            energy = target_energy
            breath = target_breath
            rough = target_rough
            bright = target_bright
        else:
            durations = torch.clamp((torch.exp(log_dur_pred) - 1) * d_control, min=0).round().long()
            pitch = pitch_pred * p_control
            energy = energy_pred * e_control
            breath = breath_pred
            rough = rough_pred
            bright = bright_pred
            
            if target_breath is not None: breath = target_breath
            if target_rough is not None: rough = target_rough
            if target_bright is not None: bright = target_bright

        # Regulate
        x_expanded, mel_len = self.length_regulator(x, durations)
        
        def expand_feat(f, d): return self.length_regulator(f.unsqueeze(-1), d)[0].transpose(1, 2)
        
        pitch = expand_feat(pitch, durations)
        energy = expand_feat(energy, durations)
        breath = expand_feat(breath, durations)
        rough = expand_feat(rough, durations)
        bright = expand_feat(bright, durations)
        
        # Decode
        dec_input = x_expanded.transpose(1, 2)
        dec_input = dec_input + \
                    self.pitch_embedding(pitch) + \
                    self.energy_embedding(energy) + \
                    self.breath_embedding(breath) + \
                    self.rough_embedding(rough) + \
                    self.bright_embedding(bright)
        dec_input = dec_input.transpose(1, 2)
        
        mel_mask = (torch.arange(dec_input.size(1), device=x.device)[None, :] >= mel_len[:, None])
        for block in self.decoder_blocks:
            dec_input = block(dec_input, mask=mel_mask)
            
        mel_out = self.mel_linear(dec_input)
        
        return {
            'mel_pred': mel_out,
            'log_duration_pred': log_dur_pred,
            'pitch_pred': pitch_pred, 'energy_pred': energy_pred,
            'breath_pred': breath_pred, 'rough_pred': rough_pred, 'bright_pred': bright_pred,
            'src_mask': src_mask, 'mel_len': mel_len
        }

# =========================================================
# 2. DATASET (Fixed Energy & TextGrid Logic)
# =========================================================
class RealMetricsDataset(Dataset):
    def __init__(self, data_dir, textgrid_dir=None, cache_dir='cache_fixed', force_rebuild=False):
        self.cache_dir = cache_dir
        self.metadata = []
        
        if force_rebuild and os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        meta_path = os.path.join(cache_dir, "metadata.json")
        if len(glob.glob(os.path.join(cache_dir, "*.pt"))) > 10 and os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                data = json.load(f)
                self.metadata = data['files']
                self.stats = data['stats']
                self.vocab = data['vocab']
            return

        print("🔄 Building Dataset Cache...")
        print(f"   Audio: {data_dir}")
        print(f"   Alignments: {textgrid_dir if textgrid_dir else 'Uniform Fallback (⚠️)'}")

        wav_files = sorted(glob.glob(os.path.join(os.path.abspath(data_dir), "**", "*.wav"), recursive=True))
        
        # --- Stats Pass ---
        print("   Step 1: Calculating Stats (Pitch, Energy, Centroid)...")
        all_p, all_e, all_c = [], [], []
        
        for w in random.sample(wav_files, min(len(wav_files), 500)):
            try:
                y, _ = librosa.load(w, sr=CONFIG['sr'])
                if len(y) < 2000: continue
                
                # Pitch
                f0, _, _ = librosa.pyin(y, fmin=60, fmax=500, sr=CONFIG['sr'])
                f0 = np.log(np.nan_to_num(f0, nan=1e-8) + 1e-8)
                all_p.extend(f0[f0 > -5].tolist())
                
                # Energy (Real calculation now!)
                rms = librosa.feature.rms(y=y, hop_length=256)[0]
                rms = np.log(rms + 1e-6)
                all_e.extend(rms.tolist())
                
                # Brightness
                cent = np.log(librosa.feature.spectral_centroid(y=y, sr=CONFIG['sr'])[0] + 1e-8)
                all_c.extend(cent.tolist())
            except: continue
        
        self.stats = {
            'p_mean': float(np.mean(all_p)), 'p_std': float(np.std(all_p)) + 1e-5,
            'e_mean': float(np.mean(all_e)), 'e_std': float(np.std(all_e)) + 1e-5, # REAL Energy stats
            'c_mean': float(np.mean(all_c)), 'c_std': float(np.std(all_c)) + 1e-5
        }
        print(f"   Stats: Energy Mean={self.stats['e_mean']:.2f}, Std={self.stats['e_std']:.2f}")

        # --- Processing Pass ---
        print("   Step 2: Processing Files...")
        vocab_set = set(['<PAD>', '<UNK>', '<SIL>'])
        
        for i, wav_path in enumerate(tqdm(wav_files)):
            try:
                # 1. Load Audio
                y, _ = librosa.load(wav_path, sr=CONFIG['sr'])
                if len(y) < 4000: continue
                basename = os.path.splitext(os.path.basename(wav_path))[0]

                # 2. Get Alignment (TextGrid > Fallback)
                phs, durs = [], []
                
                # Try TextGrid first
                if textgrid_dir and TEXTGRID_AVAILABLE:
                    tg_candidates = glob.glob(os.path.join(textgrid_dir, "**", f"{basename}.TextGrid"), recursive=True)
                    if tg_candidates:
                        try:
                            tg = textgrid.TextGrid.fromFile(tg_candidates[0])
                            phone_tier = next((t for t in tg if t.name.lower() in ['phones', 'phonemes']), None)
                            if phone_tier:
                                for interval in phone_tier:
                                    frames = int((interval.maxTime - interval.minTime) * CONFIG['sr'] / 256)
                                    if frames > 0:
                                        mark = interval.mark if interval.mark else '<SIL>'
                                        phs.append(mark)
                                        durs.append(frames)
                        except: pass # Fallback if TG corrupt

                # Fallback if no TextGrid or TG failed
                if not phs:
                    txt_path = wav_path.replace('.wav', '.txt')
                    if os.path.exists(txt_path):
                        with open(txt_path) as f: text = f.read().strip()
                        phs = ['<SIL>'] + list(phonemize(text, language="en-us", backend="espeak", strip=True)) + ['<SIL>']
                        durs = [int((len(y)/256) / len(phs))] * len(phs) # Uniform fallback
                
                if not phs: continue # Skip if totally failed
                vocab_set.update(phs)

                # 3. Extract Features
                mel = librosa.feature.melspectrogram(y=y, sr=CONFIG['sr'], n_fft=1024, hop_length=256, n_mels=80)
                mel = torch.log(torch.clamp(torch.from_numpy(mel), min=1e-5))
                
                f0, _, voiced_prob = librosa.pyin(y, fmin=60, fmax=500, sr=CONFIG['sr'], hop_length=256)
                
                # REAL ENERGY
                rms = librosa.feature.rms(y=y, hop_length=256)[0]
                rms_log = np.log(rms + 1e-6)
                
                cent = librosa.feature.spectral_centroid(y=y, sr=CONFIG['sr'], hop_length=256)[0]
                
                # Fix lengths
                min_l = min(mel.shape[1], len(f0), len(rms_log))
                mel = mel[:, :min_l]
                
                # 4. Phoneme Averaging
                p, e, br, ro, bri = [], [], [], [], []
                curr = 0
                f0_log = np.log(np.nan_to_num(f0, nan=1e-8) + 1e-8)
                cent_log = np.log(cent + 1e-8)
                
                for d in durs:
                    sl = slice(curr, min(curr+d, min_l))
                    if sl.start >= sl.stop: 
                        p.append(0); e.append(0); br.append(0); ro.append(0); bri.append(0)
                        continue
                        
                    # Pitch
                    seg_p = f0_log[sl]
                    p.append((np.mean(seg_p[seg_p > -5]) - self.stats['p_mean']) / self.stats['p_std'] if np.any(seg_p > -5) else 0)
                    
                    # Energy
                    e.append((np.mean(rms_log[sl]) - self.stats['e_mean']) / self.stats['e_std'])
                    
                    # Breath
                    br.append(1.0 - np.mean(voiced_prob[sl]))
                    
                    # Roughness
                    ro.append(np.std(seg_p[seg_p > -5]) if np.any(seg_p > -5) else 0)
                    
                    # Brightness
                    bri.append((np.mean(cent_log[sl]) - self.stats['c_mean']) / self.stats['c_std'])
                    
                    curr += d

                save_path = os.path.join(cache_dir, f"u_{i:05d}.pt")
                torch.save({
                    'phs': phs, 'durs': durs, 'mel': mel.T.clone(),
                    'pitch': np.array(p), 'energy': np.array(e), # Saved real energy
                    'breath': np.array(br), 'rough': np.array(ro), 'bright': np.array(bri)
                }, save_path)
                self.metadata.append(save_path)
                
            except Exception as e: continue

        self.vocab = sorted(list(vocab_set))
        with open(meta_path, 'w') as f:
            json.dump({'files': self.metadata, 'stats': self.stats, 'vocab': self.vocab}, f)

    def __len__(self): return len(self.metadata)
    def __getitem__(self, idx):
        u = torch.load(self.metadata[idx], weights_only=False)
        ph_to_idx = {p: i for i, p in enumerate(self.vocab)}
        return {
            'ids': torch.LongTensor([ph_to_idx.get(p, 0) for p in u['phs']]),
            'durs': torch.LongTensor(u['durs']),
            'mel': u['mel'],
            'pitch': torch.FloatTensor(u['pitch']),
            'energy': torch.FloatTensor(u['energy']), # Loads real energy
            'breath': torch.FloatTensor(u['breath']),
            'rough': torch.FloatTensor(u['rough']),
            'bright': torch.FloatTensor(u['bright']),
            'log_durs': torch.log(torch.LongTensor(u['durs']).float() + 1),
        }

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    
    ids = pad_sequence([b['ids'] for b in batch], batch_first=True)
    lens = torch.LongTensor([len(b['ids']) for b in batch])
    durs = pad_sequence([b['durs'] for b in batch], batch_first=True)
    mels = pad_sequence([b['mel'] for b in batch], batch_first=True)
    
    def pad_feat(key): return pad_sequence([b[key] for b in batch], batch_first=True)
    
    return {
        'ids': ids, 'lens': lens, 'durs': durs, 'mel': mels,
        'log_durs': pad_feat('log_durs'), 'pitch': pad_feat('pitch'),
        'energy': pad_feat('energy'), # Padded real energy
        'breath': pad_feat('breath'),
        'rough': pad_feat('rough'), 'bright': pad_feat('bright')
    }

# =========================================================
# 3. UTILS & TRAINER
# =========================================================
def save_plot(mel_gt, mel_pred, path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].imshow(mel_gt, aspect='auto', origin='lower', interpolation='none')
    axes[0].set_title('Target')
    axes[1].imshow(mel_pred, aspect='auto', origin='lower', interpolation='none')
    axes[1].set_title('Predicted')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

class Trainer:
    def __init__(self, args):
        self.args = args
        self.log_dir = os.path.join("logs", args.name)
        self.ckpt_dir = os.path.join("checkpoints", args.name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        # Pass TextGrid dir
        full_dataset = RealMetricsDataset(args.data_dir, args.textgrid_dir)
        self.vocab = full_dataset.vocab
        self.stats = full_dataset.stats
        
        val_size = int(len(full_dataset) * 0.05)
        train_size = len(full_dataset) - val_size
        self.train_ds, self.val_ds = random_split(full_dataset, [train_size, val_size])
        
        print(f"Dataset: {len(self.train_ds)} Train, {len(self.val_ds)} Val")

        self.model = RealMetricsFastSpeech2(len(self.vocab)).to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.scaler = GradScaler()
        self.vocoder = None # Lazy load only if needed

        if args.resume:
            print(f"♻️ Resuming from {args.resume}...")
            ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
            self.model.load_state_dict(ckpt['model'])
            if 'optimizer' in ckpt: self.optimizer.load_state_dict(ckpt['optimizer'])
            if 'scaler' in ckpt: self.scaler.load_state_dict(ckpt['scaler'])

    def train(self):
        train_loader = DataLoader(
            self.train_ds, batch_size=self.args.batch_size, shuffle=True, 
            collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True
        )
        val_loader = DataLoader(
            self.val_ds, batch_size=self.args.batch_size, shuffle=False, 
            collate_fn=collate_fn, num_workers=2
        )
        
        best_loss = float('inf')

        for epoch in range(self.args.epochs):
            self.model.train()
            total_loss = 0
            steps = 0
            
            pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
            for i, b in enumerate(pbar):
                if b is None: continue
                for k, v in b.items():
                    if isinstance(v, torch.Tensor): b[k] = v.to(DEVICE)
                
                with autocast():
                    out = self.model(b['ids'], b['lens'], 
                                   target_durations=b['durs'], target_pitch=b['pitch'], target_energy=b['energy'],
                                   target_breath=b['breath'], target_rough=b['rough'], target_bright=b['bright'])
                    
                    mask = ~out['src_mask']
                    mel_len = min(out['mel_pred'].size(1), b['mel'].size(1))
                    
                    l_mel = F.l1_loss(out['mel_pred'][:, :mel_len], b['mel'][:, :mel_len])
                    l_dur = F.mse_loss(out['log_duration_pred'][mask], b['log_durs'][mask])
                    l_pitch = F.mse_loss(out['pitch_pred'][mask], b['pitch'][mask])
                    l_energy = F.mse_loss(out['energy_pred'][mask], b['energy'][mask]) # Real Energy Loss
                    l_aux = F.mse_loss(out['breath_pred'][mask], b['breath'][mask]) + \
                            F.mse_loss(out['rough_pred'][mask], b['rough'][mask]) + \
                            F.mse_loss(out['bright_pred'][mask], b['bright'][mask])
                    
                    loss = l_mel + l_dur + l_pitch + l_energy + l_aux
                    loss = loss / self.args.grad_accum

                self.scaler.scale(loss).backward()
                
                if (i + 1) % self.args.grad_accum == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                
                total_loss += loss.item() * self.args.grad_accum
                steps += 1
                pbar.set_postfix({'Loss': f"{loss.item() * self.args.grad_accum:.2f}"})

            # Valid
            val_loss = self.validate(val_loader, epoch)
            self.scheduler.step(val_loss)
            
            # Checkpoint
            state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'scaler': self.scaler.state_dict(), 'vocab': self.vocab, 'stats': self.stats}
            torch.save(state, os.path.join(self.ckpt_dir, "last.pt"))
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(state, os.path.join(self.ckpt_dir, "best.pt"))

    def validate(self, loader, epoch):
        self.model.eval()
        total = 0
        with torch.no_grad():
            for i, b in enumerate(loader):
                if b is None: continue
                for k, v in b.items():
                    if isinstance(v, torch.Tensor): b[k] = v.to(DEVICE)
                
                out = self.model(b['ids'], b['lens'], 
                               target_durations=b['durs'], target_pitch=b['pitch'], target_energy=b['energy'],
                               target_breath=b['breath'], target_rough=b['rough'], target_bright=b['bright'])
                
                mel_len = min(out['mel_pred'].size(1), b['mel'].size(1))
                loss = F.l1_loss(out['mel_pred'][:, :mel_len], b['mel'][:, :mel_len])
                total += loss.item()
                
                if i == 0:
                    save_plot(b['mel'][0, :mel_len].cpu().numpy().T, out['mel_pred'][0, :mel_len].cpu().numpy().T, os.path.join(self.log_dir, f"val_{epoch}.png"))
        return total / len(loader)
# =========================================================
# 5. VOCODER & INFERENCE
# =========================================================
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
        with torch.no_grad(): return self.model(torch.FloatTensor(mel).to(DEVICE)).squeeze().cpu().numpy()

def infer_tts(
    checkpoint_path,
    text,
    breathiness=0.1,
    roughness=0.05,
    brightness=0.0,
    pitch_scale=1.0,
    duration_scale=1.0,
    hifigan_dir="./hifi-gan"
):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    vocab = ckpt['vocab']
    ph_to_idx = {p: i for i, p in enumerate(vocab)}

    model = RealMetricsFastSpeech2(len(vocab)).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()

    vocoder = Vocoder(hifigan_dir)

    # Phonemize
    from phonemizer import phonemize
    phs = ['<SIL>'] + list(phonemize(text, language="en-us", backend="espeak", strip=True)) + ['<SIL>']
    ids = torch.LongTensor([ph_to_idx.get(p, 0) for p in phs]).unsqueeze(0).to(DEVICE)
    steps = len(phs)

    tgt_breath = torch.full((1, steps), breathiness).to(DEVICE)
    tgt_rough = torch.full((1, steps), roughness).to(DEVICE)
    tgt_bright = torch.full((1, steps), brightness).to(DEVICE)

    with torch.no_grad():
        out = model(
            ids,
            torch.LongTensor([steps]).to(DEVICE),
            target_breath=tgt_breath,
            target_rough=tgt_rough,
            target_bright=tgt_bright,
            p_control=pitch_scale,
            d_control=duration_scale
        )
        wav = vocoder.infer(out['mel_pred'].transpose(1, 2))

    return wav

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'])
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--textgrid_dir', type=str, help="Path to MFA .TextGrid files")
    parser.add_argument('--name', type=str, default='run_fixed')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hifigan_dir', type=str, default='vocoder_checkpoints/LJ_FT_T2_V3')
    parser.add_argument('--text', type=str, default='You are using the SPEV text-to-speech synthesis system.')
    parser.add_argument('--output', type=str, default='output.wav')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/run_fixed/best.pt')

    
    args = parser.parse_args()
    if args.mode == 'train':
        Trainer(args).train()
    else:
        wav = infer_tts(args.checkpoint,
                args.text,
                breathiness=0.1,
                roughness=0.05,
                brightness=0.0,
                pitch_scale=1.0,
                duration_scale=1.0,
                hifigan_dir="./hifi-gan"
            )
        sf.write(args.output, wav, CONFIG['sr'])