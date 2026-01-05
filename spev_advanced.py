"""
SPEV - EMBODIED CORE (The "Spinal Cord")
========================================

This is the COORDINATOR layer that sits above the TTS engine.
It solves the "Decision Maker" problem.

Architecture:
1. Brain (LLM/Logic) -> Decides Intent & Events
2. Prosody Policy    -> Maps Intent to Acoustic Knobs (Breath/Rough/Pitch)
3. Event Synth       -> Generates Non-Verbal Audio (Sighs/Breaths/Laughs) via DSP
4. SPEV TTS          -> Generates Speech (Phonemes)
5. Mixer             -> Blends them into a single biological stream

Usage:
  python spev_embodied_core.py --text "I am so tired... [sigh] but I must go on." --emotion exhausted --checkpoint checkpoints/best_model.pt
"""

import os
import sys
import torch
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
    print("   Please ensure spev_real_metrics.py is in the same folder.")
    sys.exit(1)

from phonemizer import phonemize

# =========================================================
# 1. EVENT SYNTH (The Parallel Generator)
# =========================================================
class VocalEventSynth:
    """
    Procedural Audio Generator for Non-Verbal Sounds.
    Does not require external .wav files. Uses DSP to synthesize
    biological sounds (sighs, breaths, hums) on the fly.
    """
    def __init__(self, sr=22050):
        self.sr = sr

    def generate_sigh(self, duration=1.2, intensity=0.8):
        """Generates a spectral sigh (filtered noise with decaying envelope)"""
        t = np.linspace(0, duration, int(self.sr * duration))
        
        # 1. Source: Pink Noise
        noise = np.random.randn(len(t))
        b, a = scipy.signal.butter(1, 0.2) # Lowpass roughly
        noise = scipy.signal.lfilter(b, a, noise)
        
        # 2. Envelope: Attack -> Sustain -> Slow Decay
        env = np.concatenate([
            np.linspace(0, 1, int(0.2 * self.sr)),
            np.linspace(1, 0.6, int(0.3 * self.sr)),
            np.linspace(0.6, 0, int((duration - 0.5) * self.sr))
        ])
        # Pad envelope if rounding errors
        if len(env) < len(noise):
            env = np.pad(env, (0, len(noise)-len(env)))
        else:
            env = env[:len(noise)]
            
        # 3. Formant Filter (Sigh is essentially an 'H' sound, open vocal tract)
        # Bandpass around 1000Hz - 4000Hz
        sos = scipy.signal.butter(2, [800, 4000], btype='bandpass', fs=self.sr, output='sos')
        filtered = scipy.signal.sosfilt(sos, noise)
        
        return filtered * env * intensity * 0.15

    def generate_breath_in(self, duration=0.4, intensity=0.6):
        """Generates a sharp intake of breath"""
        t = np.linspace(0, duration, int(self.sr * duration))
        noise = np.random.randn(len(t))
        
        # Envelope: Fast Attack -> Sharp Cutoff
        env = np.linspace(0, 1, len(t)) ** 2  # Exponential rise
        
        # Filter: Higher frequency for intake
        sos = scipy.signal.butter(2, [1500, 6000], btype='bandpass', fs=self.sr, output='sos')
        filtered = scipy.signal.sosfilt(sos, noise)
        
        return filtered * env * intensity * 0.1

    def generate_grunt(self, duration=0.2, intensity=0.5):
        """Generates a laryngeal grunt (creaky impulse train)"""
        t = np.linspace(0, duration, int(self.sr * duration))
        
        # Impulse train at low freq (e.g., 60Hz)
        f0 = 60
        pulses = (np.sin(2 * np.pi * f0 * t) > 0.95).astype(float)
        
        # Add jitter
        jitter = np.random.randn(len(t)) * 0.1
        pulses += jitter
        
        # Envelope: Bell curve
        env = np.exp(-((t - duration/2)**2) / 0.005)
        
        return pulses * env * intensity * 0.2

    def get_event(self, event_name):
        if 'sigh' in event_name: return self.generate_sigh()
        if 'breath' in event_name: return self.generate_breath_in()
        if 'grunt' in event_name: return self.generate_grunt()
        return np.zeros(100)

# =========================================================
# 2. PROSODY POLICY (The Brain/Decision Maker)
# =========================================================
class ProsodyPolicy:
    """
    Translates High-Level Intent (Emotion) -> Low-Level Acoustic Knobs.
    This is the mapping layer you were missing.
    """
    def __init__(self):
        # Default neutral state
        self.default_style = {
            'breathiness': 0.1,
            'roughness': 0.05,
            'brightness': 0.0,
            'pitch_scale': 1.0,
            'duration_scale': 1.0
        }
        
        # The Rules Engine
        self.styles = {
            'neutral': self.default_style,
            
            'exhausted': {
                'breathiness': 0.7,   # Very breathy
                'roughness': 0.4,     # Creaky voice
                'brightness': -1.0,   # Muffled
                'pitch_scale': 0.8,   # Low pitch
                'duration_scale': 1.2 # Slow speech
            },
            
            'excited': {
                'breathiness': 0.0,   # Clear voice
                'roughness': 0.0,     # Stable
                'brightness': 1.5,    # Bright/Forward
                'pitch_scale': 1.3,   # High pitch
                'duration_scale': 0.9 # Fast speech
            },
            
            'secretive': {
                'breathiness': 0.9,   # Whisper
                'roughness': 0.0,
                'brightness': -0.5,
                'pitch_scale': 1.0,
                'duration_scale': 1.1
            },
            
            'angry': {
                'breathiness': 0.0,   # Tense/Pressed
                'roughness': 0.6,     # Growl
                'brightness': 1.0,    # Sharp
                'pitch_scale': 1.1,
                'duration_scale': 0.8 # Fast/Sharp
            }
        }

    def get_knobs(self, emotion):
        return self.styles.get(emotion, self.default_style)

# =========================================================
# 3. EMBODIED AGENT (The Spinal Cord / Coordinator)
# =========================================================
class EmbodiedAgent:
    def __init__(self, checkpoint_path, hifigan_dir):
        print("🧠 Initializing Spinal Cord...")
        
        # 1. Load TTS Muscle
        print("   💪 Loading SPEV TTS Model...")
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        self.vocab = ckpt['vocab']
        self.ph_to_idx = {p: i for i, p in enumerate(self.vocab)}
        
        self.tts_model = RealMetricsFastSpeech2(len(self.vocab)).to(DEVICE)
        self.tts_model.load_state_dict(ckpt['model'])
        self.tts_model.eval()
        
        self.vocoder = Vocoder(hifigan_dir)
        
        # 2. Load Event Generator
        print("   🫁 Loading Event Synthesizer...")
        self.event_synth = VocalEventSynth(sr=CONFIG['sr'])
        
        # 3. Load Policy
        self.policy = ProsodyPolicy()
        print("✅ System Ready.\n")

    def synthesize(self, text_input, emotion="neutral"):
        """
        Takes raw text with event tags (e.g. "Hello [sigh] world")
        and orchestrates the generation.
        """
        # 1. Get Style Knobs based on Emotion
        knobs = self.policy.get_knobs(emotion)
        print(f"🎭 Emotion: {emotion.upper()}")
        print(f"   Knobs: Breath={knobs['breathiness']}, Rough={knobs['roughness']}, Bright={knobs['brightness']}")

        # 2. Parse Text for Events
        # Regex to split "[event]" from text
        tokens = re.split(r'(\[.*?\])', text_input)
        tokens = [t.strip() for t in tokens if t.strip()]
        
        audio_segments = []
        
        for token in tokens:
            if token.startswith('[') and token.endswith(']'):
                # --- EVENT PATH ---
                event_name = token[1:-1].lower()
                print(f"   ⚡ Triggering Event: {event_name}")
                audio_chunk = self.event_synth.get_event(event_name)
                audio_segments.append(audio_chunk)
                # Add a small silence after event
                audio_segments.append(np.zeros(int(CONFIG['sr'] * 0.1)))
                
            else:
                # --- SPEECH PATH ---
                print(f"   🗣️  Speaking: '{token}'")
                
                # Phonemize
                phs = ['<SIL>'] + list(phonemize(token, language="en-us", backend="espeak", strip=True)) + ['<SIL>']
                ids = torch.LongTensor([self.ph_to_idx.get(p, 0) for p in phs]).unsqueeze(0).to(DEVICE)
                
                # Create Control Tensors from Policy Knobs
                tgt_breath = torch.full((1, len(phs)), knobs['breathiness']).to(DEVICE)
                tgt_rough = torch.full((1, len(phs)), knobs['roughness']).to(DEVICE)
                tgt_bright = torch.full((1, len(phs)), knobs['brightness']).to(DEVICE)
                
                with torch.no_grad():
                    out = self.tts_model(
                        ids, 
                        torch.LongTensor([len(phs)]).to(DEVICE),
                        target_breath=tgt_breath,
                        target_rough=tgt_rough,
                        target_bright=tgt_bright,
                        p_control=knobs['pitch_scale'],
                        d_control=knobs['duration_scale']
                    )
                    wav = self.vocoder.infer(out['mel_pred'].transpose(1, 2))
                    audio_segments.append(wav)
                    
        # 3. Mixer (Concatenate for now, crossfade is better but complex)
        full_audio = np.concatenate(audio_segments)
        return full_audio

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True, help="Text with events, e.g. 'Hi [sigh] bye'")
    parser.add_argument('--emotion', type=str, default='neutral', choices=['neutral', 'exhausted', 'excited', 'secretive', 'angry'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hifigan_dir', type=str, default='./hifi-gan')
    parser.add_argument('--output', type=str, default='embodied_output.wav')
    
    args = parser.parse_args()
    
    agent = EmbodiedAgent(args.checkpoint, args.hifigan_dir)
    audio = agent.synthesize(args.text, args.emotion)
    
    sf.write(args.output, audio, CONFIG['sr'])
    print(f"\n💾 Output saved to {args.output}")