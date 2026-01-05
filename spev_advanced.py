"""
SPEV - ADVANCED VOICE CONTROL EDITION
======================================

New Features:
1. Voice Quality Controls (Breathiness, Roughness/Creak, Nasality)
2. Emotional Intensity (VAD: Valence-Arousal-Dominance)
3. Word-Level Focus/Emphasis Control
4. Physiological Constraints (Age, Lung Capacity/Breath Phrasing)

Usage:
  Train:
  python spev_advanced.py --mode train --data_dir ./wavs --textgrid_dir ./aligned --hifigan_dir ./hifi-gan

  Inference with Advanced Controls:
  python spev_advanced.py --mode infer --text "Hello world." --checkpoint checkpoints/best_model.pt \
    --breathiness 0.3 --roughness 0.1 --nasality 0.0 \
    --valence 0.8 --arousal 0.6 --dominance 0.5 \
    --age 35 --lung_capacity 0.8 \
    --word_emphasis "1.0,1.0,2.5,1.0"
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
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
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.append('hifi-gan')

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
# 1. ADVANCED MODEL ARCHITECTURE
# =========================================================

def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = lengths.max()
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1))
    return ~mask

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

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

class BreathGenerator(nn.Module):
    """Generates breath insertion points based on lung capacity"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, lung_capacity=1.0):
        """Returns probability of needing breath at each position"""
        breath_need = self.predictor(x).squeeze(-1)
        # Lower lung capacity = more frequent breaths
        breath_need = breath_need * (2.0 - lung_capacity)
        return breath_need

class AdvancedFastSpeech2(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=256, n_mels=80):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = SinusoidalPositionalEmbedding(hidden_dim)
        
        self.encoder_blocks = nn.ModuleList([FFTBlock(hidden_dim) for _ in range(4)])
        
        # Standard predictors
        self.duration_predictor = VariancePredictor(hidden_dim)
        self.pitch_predictor = VariancePredictor(hidden_dim)
        self.energy_predictor = VariancePredictor(hidden_dim)
        
        # NEW: Voice Quality Predictors
        self.breathiness_predictor = VariancePredictor(hidden_dim)
        self.roughness_predictor = VariancePredictor(hidden_dim)
        self.nasality_predictor = VariancePredictor(hidden_dim)
        
        # NEW: VAD Emotion Embedding (3D spherical space)
        self.vad_projection = nn.Linear(3, hidden_dim)  # valence, arousal, dominance
        
        # NEW: Age scaling factor
        self.age_embedding = nn.Embedding(100, hidden_dim)  # ages 0-99
        
        # NEW: Breath generator
        self.breath_generator = BreathGenerator(hidden_dim)
        
        # Bucketed embeddings
        self.pitch_bins = nn.Parameter(torch.linspace(-3, 3, 256), requires_grad=False)
        self.pitch_embedding = nn.Embedding(256, hidden_dim)
        
        self.energy_bins = nn.Parameter(torch.linspace(-3, 3, 256), requires_grad=False)
        self.energy_embedding = nn.Embedding(256, hidden_dim)
        
        # NEW: Voice quality embeddings
        self.breathiness_bins = nn.Parameter(torch.linspace(0, 1, 64), requires_grad=False)
        self.breathiness_embedding = nn.Embedding(64, hidden_dim)
        
        self.roughness_bins = nn.Parameter(torch.linspace(0, 1, 64), requires_grad=False)
        self.roughness_embedding = nn.Embedding(64, hidden_dim)
        
        self.nasality_bins = nn.Parameter(torch.linspace(0, 1, 64), requires_grad=False)
        self.nasality_embedding = nn.Embedding(64, hidden_dim)
        
        self.length_regulator = LengthRegulator()
        self.decoder_blocks = nn.ModuleList([FFTBlock(hidden_dim) for _ in range(4)])
        self.mel_linear = nn.Linear(hidden_dim, n_mels)
        
        # NEW: Voice quality output heads (modulates mel output)
        self.spectral_modulator = nn.Linear(hidden_dim, n_mels)

    def apply_word_emphasis(self, x, emphasis_weights):
        """
        Apply word-level emphasis by scaling duration/pitch/energy
        emphasis_weights: [batch, seq_len] tensor of scaling factors
        """
        if emphasis_weights is not None:
            emphasis_weights = emphasis_weights.unsqueeze(-1)  # [B, T, 1]
            x = x * (0.5 + emphasis_weights * 0.5)  # Scale encoding
        return x

    def apply_age_scaling(self, pitch_pred, age):
        """
        Modify pitch based on age (vocal fold thickness)
        Young: higher pitch, Old: lower pitch
        """
        # Age 0-20: +20% pitch, Age 50+: -20% pitch, Age 25 (mid): 0%
        age_factor = 1.0 + (25 - age) * 0.008  # Linear scaling
        return pitch_pred * age_factor

    def forward(self, phoneme_ids, lengths, 
                target_durations=None, target_pitch=None, target_energy=None,
                target_breathiness=None, target_roughness=None, target_nasality=None,
                vad_vector=None,  # [batch, 3] for valence/arousal/dominance
                age=None,  # [batch] age values
                word_emphasis=None,  # [batch, seq_len] emphasis weights
                lung_capacity=1.0,
                d_control=1.0, p_control=1.0, e_control=1.0):
        
        x = self.embedding(phoneme_ids)
        x = self.pos_encoder(x)
        
        # Apply VAD emotion if provided
        if vad_vector is not None:
            vad_emb = self.vad_projection(vad_vector).unsqueeze(1)  # [B, 1, hidden]
            x = x + vad_emb
        
        # Apply age embedding if provided
        if age is not None:
            age_emb = self.age_embedding(age).unsqueeze(1)  # [B, 1, hidden]
            x = x + age_emb
        
        src_mask = get_mask_from_lengths(lengths, max_len=x.size(1))
        for block in self.encoder_blocks:
            x = block(x, mask=src_mask)
        enc_out = x
        
        # Apply word-level emphasis
        if word_emphasis is not None:
            enc_out = self.apply_word_emphasis(enc_out, word_emphasis)
        
        # Predict variance features
        log_dur_pred = self.duration_predictor(enc_out)
        pitch_pred = self.pitch_predictor(enc_out)
        energy_pred = self.energy_predictor(enc_out)
        
        # NEW: Predict voice quality
        breathiness_pred = torch.sigmoid(self.breathiness_predictor(enc_out))
        roughness_pred = torch.sigmoid(self.roughness_predictor(enc_out))
        nasality_pred = torch.sigmoid(self.nasality_predictor(enc_out))
        
        # Apply age scaling to pitch
        if age is not None:
            pitch_pred = self.apply_age_scaling(pitch_pred, age.float().mean())
        
        # Use targets or predictions
        if target_durations is not None:
            durations = target_durations
            pitch_vals = target_pitch
            energy_vals = target_energy
            breathiness_vals = target_breathiness if target_breathiness is not None else breathiness_pred
            roughness_vals = target_roughness if target_roughness is not None else roughness_pred
            nasality_vals = target_nasality if target_nasality is not None else nasality_pred
        else:
            durations = torch.clamp((torch.exp(log_dur_pred) - 1) * d_control, min=0).round().long()
            pitch_vals = pitch_pred * p_control
            energy_vals = energy_pred * e_control
            breathiness_vals = breathiness_pred
            roughness_vals = roughness_pred
            nasality_vals = nasality_pred
        
        # Apply breath insertion based on lung capacity
        breath_need = self.breath_generator(enc_out, lung_capacity)
        # Insert micro-pauses where breath_need > threshold
        breath_mask = (breath_need > 0.7).float()
        durations = durations + (breath_mask * 2).long()  # Add 2 frames for breath
        
        # Clamp and bucketize
        pitch_vals = torch.clamp(pitch_vals, min=-3.0, max=3.0)
        energy_vals = torch.clamp(energy_vals, min=-3.0, max=3.0)
        breathiness_vals = torch.clamp(breathiness_vals, min=0.0, max=1.0)
        roughness_vals = torch.clamp(roughness_vals, min=0.0, max=1.0)
        nasality_vals = torch.clamp(nasality_vals, min=0.0, max=1.0)
        
        p_idx = torch.bucketize(pitch_vals, self.pitch_bins)
        e_idx = torch.bucketize(energy_vals, self.energy_bins)
        b_idx = torch.bucketize(breathiness_vals, self.breathiness_bins)
        r_idx = torch.bucketize(roughness_vals, self.roughness_bins)
        n_idx = torch.bucketize(nasality_vals, self.nasality_bins)
        
        # Combine all embeddings
        x_adapted = (enc_out + 
                     self.pitch_embedding(p_idx) + 
                     self.energy_embedding(e_idx) +
                     self.breathiness_embedding(b_idx) +
                     self.roughness_embedding(r_idx) +
                     self.nasality_embedding(n_idx))
        
        x_expanded, mel_len = self.length_regulator(x_adapted, durations)
        x_expanded = self.pos_encoder(x_expanded)
        
        mel_mask = get_mask_from_lengths(mel_len, max_len=x_expanded.size(1))
        
        for block in self.decoder_blocks:
            x_expanded = block(x_expanded, mask=mel_mask)
        
        mel_out = self.mel_linear(x_expanded)
        
        # NEW: Apply spectral modulation for voice quality
        spectral_mod = torch.tanh(self.spectral_modulator(x_expanded))
        mel_out = mel_out + spectral_mod * 0.3  # Subtle modulation
        
        return {
            'mel_pred': mel_out,
            'log_duration_pred': log_dur_pred,
            'pitch_pred': pitch_pred,
            'energy_pred': energy_pred,
            'breathiness_pred': breathiness_pred,
            'roughness_pred': roughness_pred,
            'nasality_pred': nasality_pred,
            'breath_need': breath_need,
            'src_mask': src_mask,
            'mel_len': mel_len
        }

# =========================================================
# 2. DATASET (Same as before)
# =========================================================
class ProperDataset(Dataset):
    def __init__(self, data_dir, textgrid_dir=None, cache_file='proper_cache_strict.pt'):
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
        
        if not wav_files:
            print("❌ No wav files found.")
            sys.exit(1)

        print("🔍 Pass 1: Scanning Vocabulary & Stats...")
        vocab_set = set(['<PAD>', '<UNK>', '<SIL>'])
        all_p, all_e = [], []
        
        for w in wav_files[:500]: 
            try:
                y, _ = librosa.load(w, sr=CONFIG['sr'])
                f0, _, _ = librosa.pyin(y, fmin=60, fmax=500, sr=CONFIG['sr'])
                f0 = np.log(np.nan_to_num(f0, nan=1e-8) + 1e-8)
                rms = np.log(librosa.feature.rms(y=y)[0] + 1e-8)
                all_p.extend(f0[f0 > -5].tolist())
                all_e.extend(rms.tolist())
            except: continue

        self.stats = {
            'p_mean': np.mean(all_p), 'p_std': np.std(all_p),
            'e_mean': np.mean(all_e), 'e_std': np.std(all_e)
        }

        print(f"⏳ Pass 2: Processing {len(wav_files)} files...")
        silence_phones = {'sil', 'sp', 'spn', '<eps>', ''}
        
        for wav_path in wav_files:
            basename = os.path.splitext(os.path.basename(wav_path))[0]
            
            tg_path = None
            if self.textgrid_dir:
                tg_candidates = glob.glob(os.path.join(self.textgrid_dir, "**", f"{basename}.TextGrid"), recursive=True)
                if tg_candidates: tg_path = tg_candidates[0]

            try:
                utt = self._process(wav_path, tg_path, silence_phones)
                if utt:
                    self.utterances.append(utt)
                    vocab_set.update(utt['phs'])
            except Exception as e:
                pass

        self.vocab = sorted(list(vocab_set))
        self.ph_to_idx = {ph: i for i, ph in enumerate(self.vocab)}
        
        torch.save({'utterances': self.utterances, 'stats': self.stats, 'vocab': self.vocab}, cache_file)
        print(f"✓ Cache saved. Vocab size: {len(self.vocab)}")

    def _process(self, wav_path, tg_path, silence_phones):
        if not tg_path or not TEXTGRID_AVAILABLE:
            return None

        y, _ = librosa.load(wav_path, sr=CONFIG['sr'])
        tg = textgrid.TextGrid.fromFile(tg_path)
        phone_tier = next((t for t in tg if t.name.lower() in ['phones', 'phonemes']), None)
        
        if not phone_tier: return None

        phs, durs = [], []
        for interval in phone_tier:
            phone = interval.mark.strip()
            if phone.lower() in silence_phones: phone = '<SIL>'
            
            duration_sec = interval.maxTime - interval.minTime
            frames = int(duration_sec * CONFIG['sr'] / CONFIG['hop_length'])
            
            if frames >= 0:
                phs.append(phone)
                durs.append(frames)
        
        if not phs: return None

        mel = librosa.feature.melspectrogram(y=y, sr=CONFIG['sr'], n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length'], n_mels=CONFIG['n_mels'])
        mel = torch.log(torch.clamp(torch.from_numpy(mel), min=1e-5))
        
        f0, _, _ = librosa.pyin(y, fmin=60, fmax=500, sr=CONFIG['sr'], hop_length=CONFIG['hop_length'])
        f0 = np.nan_to_num(f0, nan=0.0)
        rms = librosa.feature.rms(y=y, hop_length=CONFIG['hop_length'])[0]
        
        min_l = min(mel.shape[1], len(f0), len(rms), sum(durs))
        mel = mel[:, :min_l]
        f0 = f0[:min_l]
        rms = rms[:min_l]
        
        phone_pitch, phone_energy = [], []
        curr_idx = 0
        
        for d in durs:
            if d == 0:
                phone_pitch.append(0)
                phone_energy.append(0)
                continue
            
            f0_seg = f0[curr_idx : curr_idx+d]
            rms_seg = rms[curr_idx : curr_idx+d]
            
            f0_nonzero = f0_seg[f0_seg > 0]
            p_mean = np.mean(np.log(f0_nonzero + 1e-8)) if len(f0_nonzero) > 0 else 0
            p_norm = (p_mean - self.stats['p_mean']) / self.stats['p_std']
            
            e_mean = np.mean(np.log(rms_seg + 1e-8))
            e_norm = (e_mean - self.stats['e_mean']) / self.stats['e_std']
            
            phone_pitch.append(p_norm)
            phone_energy.append(e_norm)
            curr_idx += d

        return {
            'phs': phs, 'durs': durs, 'mel': mel.T.numpy(), 
            'pitch_phone': np.array(phone_pitch), 'energy_phone': np.array(phone_energy)
        }

    def __len__(self): return len(self.utterances)
    
    def __getitem__(self, idx):
        u = self.utterances[idx]
        ph_ids = torch.LongTensor([self.ph_to_idx.get(p, self.ph_to_idx.get('<UNK>', 0)) for p in u['phs']])
        durs = torch.LongTensor(u['durs'])
        return {
            'ids': ph_ids, 'durs': durs, 'log_durs': torch.log(durs.float() + 1),
            'mel': torch.FloatTensor(u['mel']), 'pitch': torch.FloatTensor(u['pitch_phone']), 
            'energy': torch.FloatTensor(u['energy_phone'])
        }

def collate_fn(batch):
    ids = pad_sequence([b['ids'] for b in batch], batch_first=True)
    lens = torch.LongTensor([len(b['ids']) for b in batch])
    durs = pad_sequence([b['durs'] for b in batch], batch_first=True)
    l_durs = pad_sequence([b['log_durs'] for b in batch], batch_first=True)
    mels = pad_sequence([b['mel'] for b in batch], batch_first=True)
    ptch = pad_sequence([b['pitch'] for b in batch], batch_first=True)
    egy = pad_sequence([b['energy'] for b in batch], batch_first=True)
    return {'ids': ids, 'lens': lens, 'durs': durs, 'log_durs': l_durs, 'mel': mels, 'pitch': ptch, 'energy': egy}

# =========================================================
# 3. VOCODER
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
            print(f"🔊 Loading HiFi-GAN: {ckpt_path}")
            with open(config_path) as f: json_config = json.loads(f.read())
            self.h = AttrDict(json_config)
            self.model = HiFiGenerator(self.h).to(self.device)
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device)['generator'])
            self.model.eval()
            self.model.remove_weight_norm()

    def infer(self, mel):
        if self.model is None:
            mel_lin = torch.exp(mel).squeeze().cpu().numpy()
            try:
                linear_spec = librosa.feature.inverse.mel_to_stft(
                    mel_lin, sr=CONFIG['sr'], n_fft=CONFIG['n_fft'], power=1.0
                )
                return librosa.griffinlim(linear_spec, n_iter=32, hop_length=CONFIG['hop_length'], win_length=CONFIG['n_fft'])
            except:
                return librosa.griffinlim(mel_lin, n_iter=32)
        
        with torch.no_grad():
            return self.model(mel).squeeze().cpu().numpy()

# =========================================================
# 4. TRAINING & INFERENCE
# =========================================================
class Trainer:
    def __init__(self, args):
        self.dataset = ProperDataset(args.data_dir, args.textgrid_dir)
        self.model = AdvancedFastSpeech2(len(self.dataset.vocab)).to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        self.vocoder = Vocoder(args.hifigan_dir) if args.hifigan_dir else None
        self.warmup_epochs = args.warmup_epochs

    def train(self, epochs=100):
        loader = DataLoader(self.dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            if epoch < self.warmup_epochs:
                var_weight = 0.0
                print(f"🔥 Warmup Phase ({epoch+1}/{self.warmup_epochs})")
            else:
                var_weight = 1.0

            for b in loader:
                self.optimizer.zero_grad()
                
                ids = b['ids'].to(DEVICE)
                lens = b['lens'].to(DEVICE)
                durs = b['durs'].to(DEVICE)
                log_durs = b['log_durs'].to(DEVICE)
                mel_targets = b['mel'].to(DEVICE)
                pitch_targets = b['pitch'].to(DEVICE)
                energy_targets = b['energy'].to(DEVICE)

                out = self.model(ids, lens, 
                               target_durations=durs,
                               target_pitch=pitch_targets,
                               target_energy=energy_targets)
                
                src_mask = ~out['src_mask']
                mel_mask_pad = get_mask_from_lengths(out['mel_len'], max_len=out['mel_pred'].size(1))
                mel_mask_valid = ~mel_mask_pad
                
                min_len = min(out['mel_pred'].size(1), mel_targets.size(1))
                mel_pred = out['mel_pred'][:, :min_len]
                mel_gt = mel_targets[:, :min_len]
                mel_mask_valid = mel_mask_valid[:, :min_len]

                loss_mel = (F.l1_loss(mel_pred, mel_gt, reduction='none') * mel_mask_valid.unsqueeze(-1)).sum() / mel_mask_valid.sum()
                
                # Duration Loss
                dur_pred = out['log_duration_pred']
                loss_dur = (F.mse_loss(dur_pred, log_durs, reduction='none') * src_mask).sum() / src_mask.sum()
                
                # Variance Loss
                loss_pitch = (F.mse_loss(out['pitch_pred'], pitch_targets, reduction='none') * src_mask).sum() / src_mask.sum()
                loss_energy = (F.mse_loss(out['energy_pred'], energy_targets, reduction='none') * src_mask).sum() / src_mask.sum()
                
                # NEW: Voice Quality Losses (only after warmup)
                if var_weight > 0:
                    # We don't have ground truth for these, so use a regularization loss
                    loss_breath = torch.mean(torch.abs(out['breathiness_pred'] - 0.3))  # Encourage moderate breathiness
                    loss_rough = torch.mean(torch.abs(out['roughness_pred'] - 0.1))  # Low roughness by default
                    loss_nasal = torch.mean(torch.abs(out['nasality_pred'] - 0.2))  # Slight nasality
                else:
                    loss_breath = loss_rough = loss_nasal = 0.0
                
                loss = (loss_mel + loss_dur + 
                        (loss_pitch + loss_energy) * var_weight +
                        (loss_breath + loss_rough + loss_nasal) * var_weight * 0.1)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs}: Loss {avg_loss:.4f}")
            
            if (epoch+1) % 10 == 0:
                self.save(f"checkpoints/ckpt_{epoch+1}.pt")
                if self.vocoder:
                    self.test_inference(epoch+1)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model': self.model.state_dict(), 
            'vocab': self.dataset.vocab, 
            'stats': self.dataset.stats
        }, path)
        print(f"💾 Saved checkpoint: {path}")

    def test_inference(self, epoch):
        """Quick test synthesis during training"""
        self.model.eval()
        print(f"🎤 Testing synthesis at epoch {epoch}...")
        
        # Use first utterance from dataset for testing
        if len(self.dataset) > 0:
            sample = self.dataset[0]
            ids = sample['ids'].unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                out = self.model(ids, torch.LongTensor([len(sample['ids'])]).to(DEVICE))
                wav = self.vocoder.infer(out['mel_pred'].transpose(1, 2))
                
            test_path = f"checkpoints/test_epoch_{epoch}.wav"
            sf.write(test_path, wav, CONFIG['sr'])
            print(f"✅ Test output: {test_path}")
        
        self.model.train()

def inference_mode(args):
    print(f"🚀 Loading model from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    vocab = ckpt['vocab']
    model = AdvancedFastSpeech2(len(vocab)).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    vocoder = Vocoder(args.hifigan_dir)
    ph_to_idx = {p: i for i, p in enumerate(vocab)}
    
    print(f"\n{'='*60}")
    print(f"📝 Text: {args.text}")
    print(f"{'='*60}")
    print(f"🎛️  VOICE CONTROLS:")
    print(f"   Breathiness: {args.breathiness:.2f} | Roughness: {args.roughness:.2f} | Nasality: {args.nasality:.2f}")
    print(f"   Age: {args.age} | Lung Capacity: {args.lung_capacity:.2f}")
    print(f"   VAD: Valence={args.valence:.2f}, Arousal={args.arousal:.2f}, Dominance={args.dominance:.2f}")
    print(f"{'='*60}\n")
    
    # Process text to phonemes using CMU Dict
    try:
        import cmudict
        d = cmudict.dict()
        words = args.text.lower().replace('.', '').replace(',', '').replace('?', '').replace('!', '').split()
        phs = []
        
        for w in words:
            if w in d: 
                phs.extend(d[w][0])
            else:
                print(f"⚠️  Word '{w}' not in CMU dict. Using <SIL>.")
                phs.append('<SIL>')
        
        # Normalize to training vocab
        clean_phs = ['<SIL>']
        for p in phs:
            if p.upper() in ph_to_idx:
                clean_phs.append(p.upper())
            else:
                p_no_stress = p[:-1] if p[-1].isdigit() else p
                if p_no_stress.upper() in ph_to_idx:
                    clean_phs.append(p_no_stress.upper())
                else:
                    clean_phs.append('<SIL>')
        clean_phs.append('<SIL>')
        
    except ImportError:
        print("❌ Error: cmudict not found. Install with: pip install cmudict")
        return

    # Convert phonemes to IDs
    ids = torch.LongTensor([ph_to_idx.get(p, ph_to_idx.get('<UNK>', 0)) for p in clean_phs]).unsqueeze(0).to(DEVICE)
    
    # Parse word emphasis if provided
    word_emphasis = None
    if args.word_emphasis:
        try:
            emphasis_vals = [float(x) for x in args.word_emphasis.split(',')]
            # Map word-level emphasis to phoneme-level (approximate)
            # For simplicity, repeat each emphasis value across ~3 phonemes per word
            phoneme_emphasis = []
            phonemes_per_word = max(1, len(clean_phs) // len(emphasis_vals))
            for val in emphasis_vals:
                phoneme_emphasis.extend([val] * phonemes_per_word)
            # Pad or trim to match phoneme count
            while len(phoneme_emphasis) < len(clean_phs):
                phoneme_emphasis.append(1.0)
            phoneme_emphasis = phoneme_emphasis[:len(clean_phs)]
            word_emphasis = torch.FloatTensor(phoneme_emphasis).unsqueeze(0).to(DEVICE)
            print(f"✨ Word emphasis applied: {args.word_emphasis}")
        except Exception as e:
            print(f"⚠️  Could not parse word emphasis: {e}")
    
    # Create VAD vector
    vad_vector = torch.FloatTensor([[args.valence, args.arousal, args.dominance]]).to(DEVICE)
    
    # Create age tensor
    age_tensor = torch.LongTensor([args.age]).to(DEVICE)
    
    # Synthesize
    with torch.no_grad():
        out = model(
            ids, 
            torch.LongTensor([len(clean_phs)]).to(DEVICE),
            vad_vector=vad_vector,
            age=age_tensor,
            word_emphasis=word_emphasis,
            lung_capacity=args.lung_capacity,
            d_control=args.duration_scale,
            p_control=args.pitch_scale
        )
        
        # Apply voice quality controls in post-processing
        mel = out['mel_pred'].clone()
        
        # Breathiness: Add noise to higher frequencies
        if args.breathiness > 0:
            noise = torch.randn_like(mel) * args.breathiness * 0.1
            mel[:, :, 40:] += noise[:, :, 40:]  # Only high freqs
        
        # Roughness: Add periodic perturbations
        if args.roughness > 0:
            creak_pattern = torch.sin(torch.linspace(0, 20*3.14159, mel.shape[1])).to(DEVICE)
            mel[:, :, :20] += creak_pattern.unsqueeze(0).unsqueeze(-1) * args.roughness * 0.15
        
        # Nasality: Boost mid frequencies, attenuate highs
        if args.nasality > 0:
            nasal_boost = torch.ones_like(mel)
            nasal_boost[:, :, 20:40] *= (1.0 + args.nasality * 0.2)
            nasal_boost[:, :, 50:] *= (1.0 - args.nasality * 0.15)
            mel = mel * nasal_boost
        
        # Vocoder synthesis
        wav = vocoder.infer(mel.transpose(1, 2))
        
    # Save output
    output_path = args.output if args.output else "output_advanced.wav"
    sf.write(output_path, wav, CONFIG['sr'])
    print(f"\n✅ Synthesis complete! Saved to: {output_path}")
    print(f"   Duration: {len(wav)/CONFIG['sr']:.2f}s")
    print(f"   Phonemes: {' '.join(clean_phs)}\n")

# =========================================================
# 5. MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(description='SPEV TTS with Advanced Voice Controls')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'], help='train or infer')
    
    # Training args
    parser.add_argument('--data_dir', type=str, help='Directory with wav files')
    parser.add_argument('--textgrid_dir', type=str, help='Directory with TextGrid alignments')
    parser.add_argument('--hifigan_dir', type=str, default='./hifi-gan', help='HiFi-GAN checkpoint directory')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='Epochs to train duration only')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs')
    
    # Inference args
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    parser.add_argument('--text', type=str, help='Text to synthesize')
    parser.add_argument('--output', type=str, help='Output wav file path')
    
    # Standard controls
    parser.add_argument('--duration_scale', type=float, default=1.0, help='Duration control')
    parser.add_argument('--pitch_scale', type=float, default=1.0, help='Pitch control')
    
    # NEW: Voice Quality Controls
    parser.add_argument('--breathiness', type=float, default=0.0, help='Breathiness (0.0-1.0)')
    parser.add_argument('--roughness', type=float, default=0.0, help='Roughness/Creak (0.0-1.0)')
    parser.add_argument('--nasality', type=float, default=0.0, help='Nasality (0.0-1.0)')
    
    # NEW: Emotional Controls (VAD)
    parser.add_argument('--valence', type=float, default=0.5, help='Valence: negative(0) to positive(1)')
    parser.add_argument('--arousal', type=float, default=0.5, help='Arousal: calm(0) to excited(1)')
    parser.add_argument('--dominance', type=float, default=0.5, help='Dominance: submissive(0) to dominant(1)')
    
    # NEW: Physiological Controls
    parser.add_argument('--age', type=int, default=30, help='Speaker age (0-99)')
    parser.add_argument('--lung_capacity', type=float, default=1.0, help='Lung capacity: low(0.3) to high(1.0)')
    
    # NEW: Word-Level Control
    parser.add_argument('--word_emphasis', type=str, help='Comma-separated emphasis weights per word (e.g., "1.0,1.0,2.5,1.0")')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not args.data_dir:
            print("❌ --data_dir required for training")
            return
        trainer = Trainer(args)
        trainer.train(epochs=args.epochs)
    
    elif args.mode == 'infer':
        if not args.checkpoint or not args.text:
            print("❌ --checkpoint and --text required for inference")
            return
        inference_mode(args)

if __name__ == "__main__":
    main()