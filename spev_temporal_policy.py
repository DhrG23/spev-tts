"""
SPEV - TEMPORAL POLICY & EMBODIED CORE
======================================

This is the UPGRADED Coordinator (Spinal Cord).
It introduces two critical features:

1. TEMPORAL CONTROL: 
   - Controls are no longer static scalars. 
   - We generate CURVES (Trajectories) for Breathiness, Roughness, and Brightness.
   - Example: "Relief" = Breathiness starts high (1.0) and fades to (0.0).

2. POLICY LEARNER (Architecture Ready):
   - Contains the `AcousticPolicyModel` class (LSTM).
   - Currently runs in 'heuristic' mode, but is ready to be trained to map 
     Phonemes -> Acoustic Curves directly.

Usage:
  python spev_temporal_policy.py --text "Oh my god, I am so relieved." --emotion relief --checkpoint checkpoints/best_model.pt
"""

import os
import sys
import os, sys
sys.path.append('hifi-gan')
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
import scipy.signal
import soundfile as sf
import argparse
import re

# Import the Muscle (The TTS Engine)
try:
    from spev_real_metrics import RealMetricsFastSpeech2, Vocoder, CONFIG, DEVICE
except ImportError:
    print("❌ Critical Error: spev_real_metrics.py not found.")
    sys.exit(1)

from phonemizer import phonemize

# =========================================================
# 1. CURVE GENERATOR (The Temporal Engine)
# =========================================================
class CurveGenerator:
    """Generates 1D temporal trajectories for acoustic controls."""
    
    @staticmethod
    def linear(start, end, steps):
        return np.linspace(start, end, steps)
    
    @staticmethod
    def constant(val, steps):
        return np.full(steps, val)
    
    @staticmethod
    def bell(peak, steps):
        t = np.linspace(-1, 1, steps)
        return peak * np.exp(-5 * t**2)

    @staticmethod
    def oscillator(base, amp, freq, steps):
        """Adds a sine wave (tremolo/vibrato effect) to a base value"""
        t = np.linspace(0, freq * 2 * np.pi, steps)
        return base + amp * np.sin(t)

# =========================================================
# 2. THE POLICY LEARNER (LSTM / SLM Architecture)
# =========================================================
class AcousticPolicyModel(nn.Module):
    """
    The 'Brain' that will eventually replace the rule-based dictionary.
    
    Future Training Objective:
    Input: Phoneme Embeddings
    Output: Frame-level (or Phone-level) curves for Breath, Rough, Bright.
    """
    def __init__(self, vocab_size, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        
        # Heads for each acoustic knob
        self.head_breath = nn.Linear(hidden_dim * 2, 1)
        self.head_rough = nn.Linear(hidden_dim * 2, 1)
        self.head_bright = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        
        # Sigmoid ensures 0-1 range for breath/rough
        breath = torch.sigmoid(self.head_breath(x)) 
        rough = torch.sigmoid(self.head_rough(x))
        # Tanh allows -1 to 1 range for brightness
        bright = torch.tanh(self.head_bright(x)) * 2.0 
        
        return breath, rough, bright

# =========================================================
# 3. PROSODY MANAGER (Rules -> Curves)
# =========================================================
class ProsodyManager:
    """
    Manages the transition from High-Level Intent to Low-Level Curves.
    Currently uses 'Heuristic Mode', but can load the LSTM above.
    """
    def __init__(self):
        # Definitions of temporal behaviors
        self.styles = {
            'neutral': {
                'breath': ('constant', 0.1),
                'rough': ('constant', 0.05),
                'bright': ('constant', 0.0),
                'pitch': 1.0, 'speed': 1.0
            },
            'exhausted': {
                'breath': ('constant', 0.8), # Always whispery
                'rough': ('linear', 0.2, 0.6), # Gets creakier at end
                'bright': ('constant', -1.5), # Muffled
                'pitch': 0.8, 'speed': 1.2
            },
            'relief': {
                'breath': ('linear', 0.9, 0.0), # Big sigh out -> clear
                'rough': ('constant', 0.0),
                'bright': ('linear', -1.0, 0.5), # Dark -> Bright
                'pitch': 0.9, 'speed': 1.1
            },
            'anxious': {
                'breath': ('oscillator', 0.3, 0.2, 3.0), # Trembling breath
                'rough': ('constant', 0.4), # Tense
                'bright': ('constant', 0.5),
                'pitch': 1.2, 'speed': 0.9
            },
            'angry': {
                'breath': ('constant', 0.0), # Pressed/Clear
                'rough': ('bell', 0.8), # Growl in the middle
                'bright': ('constant', 1.5), # Sharp
                'pitch': 1.1, 'speed': 0.85
            }
        }

    def get_curves(self, emotion, steps):
        """
        Returns numpy arrays of length `steps` for each parameter.
        """
        style = self.styles.get(emotion, self.styles['neutral'])
        
        # Helper to unpack and generate
        def generate(param_name):
            def_val = style.get(param_name, ('constant', 0.0))
            type_ = def_val[0]
            args = def_val[1:]
            
            if type_ == 'constant': return CurveGenerator.constant(args[0], steps)
            if type_ == 'linear': return CurveGenerator.linear(args[0], args[1], steps)
            if type_ == 'bell': return CurveGenerator.bell(args[0], steps)
            if type_ == 'oscillator': return CurveGenerator.oscillator(args[0], args[1], args[2], steps)
            return np.zeros(steps)

        return {
            'breath': generate('breath'),
            'rough': generate('rough'),
            'bright': generate('bright'),
            'pitch_scale': style.get('pitch', 1.0),
            'speed_scale': style.get('speed', 1.0)
        }

# =========================================================
# 4. EMBODIED AGENT (Orchestrator)
# =========================================================
class EmbodiedAgent:
    def __init__(self, checkpoint_path, hifigan_dir):
        print("🧠 Initializing Embodied Core (Temporal Edition)...")
        
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        self.vocab = ckpt['vocab']
        self.ph_to_idx = {p: i for i, p in enumerate(self.vocab)}
        
        self.tts_model = RealMetricsFastSpeech2(len(self.vocab)).to(DEVICE)
        self.tts_model.load_state_dict(ckpt['model'])
        self.tts_model.eval()
        
        self.vocoder = Vocoder(hifigan_dir)
        self.prosody_mgr = ProsodyManager()
        
        # Simple Event Synth (Sighs/Breaths from previous version)
        self.fs = CONFIG['sr']
        print("✅ System Ready.\n")

    def synthesize_event(self, event_name):
        # Procedural audio fallback (simplified for brevity)
        duration = 0.5
        if 'sigh' in event_name: duration = 1.0
        t = np.linspace(0, duration, int(self.fs * duration))
        noise = np.random.randn(len(t)) * np.exp(-3*t) * 0.1
        return noise

    def synthesize(self, text_input, emotion="neutral"):
        print(f"🎭 Emotion: {emotion.upper()} (Applying Temporal Curves)")

        # 1. Tokenize Text (Simple split by events)
        tokens = re.split(r'(\[.*?\])', text_input)
        tokens = [t.strip() for t in tokens if t.strip()]
        
        audio_segments = []
        
        for token in tokens:
            if token.startswith('[') and token.endswith(']'):
                # Event Path
                event_name = token[1:-1].lower()
                print(f"   ⚡ Event: {event_name}")
                audio_segments.append(self.synthesize_event(event_name))
                audio_segments.append(np.zeros(int(self.fs * 0.1))) # Silence
            else:
                # Speech Path
                print(f"   🗣️  Speech: '{token}'")
                phs = ['<SIL>'] + list(phonemize(token, language="en-us", backend="espeak", strip=True)) + ['<SIL>']
                steps = len(phs)
                
                # --- TEMPORAL MAGIC HAPPENS HERE ---
                # 1. Get Curves from Policy
                curves = self.prosody_mgr.get_curves(emotion, steps)
                
                # 2. Convert to Tensors
                ids = torch.LongTensor([self.ph_to_idx.get(p, 0) for p in phs]).unsqueeze(0).to(DEVICE)
                
                # Note: We unsqueeze(0) to make batch dimension
                tgt_breath = torch.FloatTensor(curves['breath']).unsqueeze(0).to(DEVICE)
                tgt_rough = torch.FloatTensor(curves['rough']).unsqueeze(0).to(DEVICE)
                tgt_bright = torch.FloatTensor(curves['bright']).unsqueeze(0).to(DEVICE)
                
                # Debug print for the user to see the curve logic working
                mid = steps // 2
                print(f"      [Curve Debug] Breath: Start={curves['breath'][0]:.2f} -> Mid={curves['breath'][mid]:.2f} -> End={curves['breath'][-1]:.2f}")
                
                with torch.no_grad():
                    out = self.tts_model(
                        ids, 
                        torch.LongTensor([steps]).to(DEVICE),
                        target_breath=tgt_breath,
                        target_rough=tgt_rough,
                        target_bright=tgt_bright,
                        p_control=curves['pitch_scale'],
                        d_control=curves['speed_scale']
                    )
                    wav = self.vocoder.infer(out['mel_pred'].transpose(1, 2))
                    audio_segments.append(wav)
                    
        return np.concatenate(audio_segments) if audio_segments else np.zeros(100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--emotion', type=str, default='neutral', 
                        choices=['neutral', 'exhausted', 'relief', 'anxious', 'angry'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hifigan_dir', type=str, default='./hifi-gan')
    parser.add_argument('--output', type=str, default='temporal_output.wav')
    
    args = parser.parse_args()
    
    agent = EmbodiedAgent(args.checkpoint, args.hifigan_dir)
    audio = agent.synthesize(args.text, args.emotion)
    
    sf.write(args.output, audio, CONFIG['sr'])
    print(f"\n💾 Output saved to {args.output}")