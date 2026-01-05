"""
SPEV - COMPLETE (Production Ready & Bug Fixed)
==============================================
Usage:
  Train:
  python spev_tts.py --mode train --data_dir ./wavs --textgrid_dir ./aligned --hifigan_dir ./hifi-gan --warmup_epochs 20

  Inference:
  python spev_tts.py --mode infer --text "Hello world." --checkpoint checkpoints/best_model.pt
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
# phonemizer removed to enforce strict ARPABET/MFA consistency
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# --- Imports & Env Setup ---
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
    print("⚠️  'textgrid' library not found. MFA support disabled. (pip install textgrid)")
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
# 1. MODEL ARCHITECTURE
# =========================================================

def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = lengths.max()
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1))
    return ~mask  # Returns True where padding exists (PyTorch MultiheadAttention convention)

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
        # mask is True for padding positions
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

class FastSpeech2Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=256, n_mels=80):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = SinusoidalPositionalEmbedding(hidden_dim)
        
        self.encoder_blocks = nn.ModuleList([FFTBlock(hidden_dim) for _ in range(4)])
        
        self.duration_predictor = VariancePredictor(hidden_dim)
        self.pitch_predictor = VariancePredictor(hidden_dim)
        self.energy_predictor = VariancePredictor(hidden_dim)
        
        # Fixed bucket range, but inputs will be clamped
        self.pitch_bins = nn.Parameter(torch.linspace(-3, 3, 256), requires_grad=False)
        self.pitch_embedding = nn.Embedding(256, hidden_dim)
        
        self.energy_bins = nn.Parameter(torch.linspace(-3, 3, 256), requires_grad=False)
        self.energy_embedding = nn.Embedding(256, hidden_dim)
        
        self.length_regulator = LengthRegulator()
        self.decoder_blocks = nn.ModuleList([FFTBlock(hidden_dim) for _ in range(4)])
        self.mel_linear = nn.Linear(hidden_dim, n_mels)

    def forward(self, phoneme_ids, lengths, target_durations=None, target_pitch=None, target_energy=None, 
                d_control=1.0, p_control=1.0, e_control=1.0):
        
        x = self.embedding(phoneme_ids)
        x = self.pos_encoder(x)
        
        src_mask = get_mask_from_lengths(lengths, max_len=x.size(1))
        for block in self.encoder_blocks:
            x = block(x, mask=src_mask)
        enc_out = x 
        
        log_dur_pred = self.duration_predictor(enc_out)
        pitch_pred = self.pitch_predictor(enc_out)
        energy_pred = self.energy_predictor(enc_out)
        
        if target_durations is not None:
            durations = target_durations
            pitch_vals = target_pitch
            energy_vals = target_energy
        else:
            durations = torch.clamp((torch.exp(log_dur_pred) - 1) * d_control, min=0).round().long()
            pitch_vals = pitch_pred * p_control
            energy_vals = energy_pred * e_control
        
        # FIX: Clamp values before bucketing to prevent index overflow
        pitch_vals = torch.clamp(pitch_vals, min=-3.0, max=3.0)
        energy_vals = torch.clamp(energy_vals, min=-3.0, max=3.0)
        
        p_idx = torch.bucketize(pitch_vals, self.pitch_bins)
        e_idx = torch.bucketize(energy_vals, self.energy_bins)
        
        x_adapted = enc_out + self.pitch_embedding(p_idx) + self.energy_embedding(e_idx)
        
        x_expanded, mel_len = self.length_regulator(x_adapted, durations)
        x_expanded = self.pos_encoder(x_expanded)
        
        # FIX: Create mask for decoder to ignore padding in expanded sequence
        mel_mask = get_mask_from_lengths(mel_len, max_len=x_expanded.size(1))

        for block in self.decoder_blocks:
            x_expanded = block(x_expanded, mask=mel_mask)
            
        mel_out = self.mel_linear(x_expanded)
        
        return {
            'mel_pred': mel_out,
            'log_duration_pred': log_dur_pred,
            'pitch_pred': pitch_pred,
            'energy_pred': energy_pred,
            'src_mask': src_mask,
            'mel_len': mel_len
        }

# =========================================================
# 2. DATASET (Strict MFA Only)
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

            # We DO NOT load text for fallback. If no TextGrid, we skip.
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
        # Strict Mode: Fail if no TextGrid
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
            
            # Keep even short frames to maintain alignment
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
            'mel': torch.FloatTensor(u['mel']), 'pitch': torch.FloatTensor(u['pitch_phone']), 'energy': torch.FloatTensor(u['energy_phone'])
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
# 3. VOCODER (Fixed GL Fallback)
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
        # mel: [B, 80, T] Log-Mel
        if self.model is None:
            # FIX: Griffin-Lim requires Linear Magnitude Spectrogram
            # 1. Log-Mel -> Mel
            mel_lin = torch.exp(mel).squeeze().cpu().numpy()
            # 2. Mel -> Linear Magnitude (Inverse Mel Basis approximation)
            try:
                # Requires librosa >= 0.7
                linear_spec = librosa.feature.inverse.mel_to_stft(
                    mel_lin, sr=CONFIG['sr'], n_fft=CONFIG['n_fft'], power=1.0
                )
                return librosa.griffinlim(linear_spec, n_iter=32, hop_length=CONFIG['hop_length'], win_length=CONFIG['n_fft'])
            except:
                print("⚠️  Warning: librosa.feature.inverse not available. GL quality will be low.")
                return librosa.griffinlim(mel_lin, n_iter=32)
        
        with torch.no_grad():
            return self.model(mel).squeeze().cpu().numpy()

class Trainer:
    def __init__(self, args):
        self.dataset = ProperDataset(args.data_dir, args.textgrid_dir)
        self.model = FastSpeech2Model(len(self.dataset.vocab)).to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        self.vocoder = Vocoder(args.hifigan_dir) if args.hifigan_dir else None
        self.warmup_epochs = args.warmup_epochs

    def train(self, epochs=100):
        loader = DataLoader(self.dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            # WARMUP SCHEDULE: Train Duration only first
            if epoch < self.warmup_epochs:
                var_weight = 0.0
                print(f"🔥 Warmup Phase ({epoch+1}/{self.warmup_epochs}): Variance prediction disabled.")
            else:
                var_weight = 1.0

            for b in loader:
                self.optimizer.zero_grad()
                
                # Move to device
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
                
                # --- Loss Calculation ---
                src_mask = ~out['src_mask']
                
                # FIX 1: Mel Loss Masking (Critical)
                # out['mel_len'] has the valid lengths from LengthRegulator
                # Mask is True for padding
                mel_mask_pad = get_mask_from_lengths(out['mel_len'], max_len=out['mel_pred'].size(1))
                mel_mask_valid = ~mel_mask_pad
                
                # Align shapes
                min_len = min(out['mel_pred'].size(1), mel_targets.size(1))
                mel_pred = out['mel_pred'][:, :min_len]
                mel_gt = mel_targets[:, :min_len]
                mel_mask_valid = mel_mask_valid[:, :min_len]

                loss_mel = (F.l1_loss(mel_pred, mel_gt, reduction='none') * mel_mask_valid.unsqueeze(-1)).sum() / mel_mask_valid.sum()
                
                # 2. Duration Loss
                dur_pred = out['log_duration_pred']
                loss_dur = (F.mse_loss(dur_pred, log_durs, reduction='none') * src_mask).sum() / src_mask.sum()
                
                # 3. Variance Loss
                loss_pitch = (F.mse_loss(out['pitch_pred'], pitch_targets, reduction='none') * src_mask).sum() / src_mask.sum()
                loss_energy = (F.mse_loss(out['energy_pred'], energy_targets, reduction='none') * src_mask).sum() / src_mask.sum()
                
                loss = loss_mel + loss_dur + (loss_pitch * var_weight) + (loss_energy * var_weight)
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.4f}")
            if (epoch+1) % 10 == 0:
                self.save(f"checkpoints/ckpt_{epoch+1}.pt")
                self.test_inference(epoch+1)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'model': self.model.state_dict(), 'vocab': self.dataset.vocab, 'stats': self.dataset.stats}, path)

    def test_inference(self, epoch):
        self.model.eval()
        # Test text - ensure it maps to vocabulary (likely ARPABET)
        # For testing, we just simulate random IDs if vocab is unknown, or use a simple mapping
        # Ideally, we would need the text processing here, but for simplicity we rely on user input in inference mode
        pass 

def inference_mode(args):
    print(f"🚀 Loading model from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    vocab = ckpt['vocab']
    model = FastSpeech2Model(len(vocab)).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    vocoder = Vocoder(args.hifigan_dir)
    ph_to_idx = {p: i for i, p in enumerate(vocab)}
    
    print(f"📝 Text: {args.text}")
    print("ℹ️  Strict Mode: Using CMU Dict (ARPABET) only.")
    
    try:
        import cmudict
        d = cmudict.dict()
        words = args.text.lower().replace('.', '').replace(',', '').replace('?', '').split()
        phs = []
        for w in words:
            if w in d: 
                phs.extend(d[w][0])
            else:
                print(f"⚠️  Word '{w}' not in CMU dict. Skipping (Strict Mode).")
        
        # Normalize to Training Vocab
        # Mappings: 'AH0' -> 'AH0', 'sil' -> '<SIL>'
        clean_phs = ['<SIL>']
        for p in phs:
             # Check exact match or uppercase
             if p.upper() in ph_to_idx:
                 clean_phs.append(p.upper())
             else:
                 # Fallback: try stripping stress if not found 'AH0' -> 'AH'
                 p_no_stress = p[:-1] if p[-1].isdigit() else p
                 if p_no_stress.upper() in ph_to_idx:
                     clean_phs.append(p_no_stress.upper())
                 else:
                     clean_phs.append('<SIL>') # Map unknown to silence rather than crash
        clean_phs.append('<SIL>')
        
    except ImportError:
        print("❌ Error: cmudict not found. Please `pip install cmudict`.")
        return

    ids = torch.LongTensor([ph_to_idx.get(p, ph_to_idx.get('<UNK>', 0)) for p in clean_phs]).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        out = model(ids, torch.LongTensor([len(clean_phs)]).to(DEVICE), d_control=args.duration_scale, p_control=args.pitch_scale)
        wav = vocoder.infer(out['mel_pred'].transpose(1, 2))
        
    sf.write("output.wav", wav, CONFIG['sr'])
    print(f"✅ Saved to output.wav")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'])
    parser.add_argument('--data_dir', type=str, default='data/training_data_ljspeech')
    parser.add_argument('--textgrid_dir', type=str, default='data/textgrid_data', help="Path to MFA TextGrids")
    parser.add_argument('--hifigan_dir', type=str, default='vocoder_checkpoints/LJ_FT_T2_V3')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--text', type=str, default="Hello world! I am using SPEV TTS. Huh It works. Maybe.")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=10, help="Epochs to train duration only")
    parser.add_argument('--duration_scale', type=float, default=1.0)
    parser.add_argument('--pitch_scale', type=float, default=1.0)
    args = parser.parse_args()
    
    if args.mode == 'train':
        Trainer(args).train(args.epochs)
    else:
        if args.checkpoint: inference_mode(args)
        else: print("❌ Checkpoint required for inference") 