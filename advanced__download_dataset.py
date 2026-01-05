"""
SPEV - UNIVERSAL DATASET PREPPER
================================
Current training code expects:
   1. A folder of .wav files
   2. A corresponding .txt file for each .wav (containing the transcript)

LJSpeech comes like this, but ESD and Jenny do not. 
This script converts ESD/Jenny into the format `spev_real_metrics.py` needs.

Usage:
  # For ESD (Emotional Speech Dataset)
  python spev_data_prep.py --dataset esd --in_dir ./ESD_English --out_dir ./training_data

  # For Jenny
  python spev_data_prep.py --dataset jenny --in_dir ./Jenny --out_dir ./training_data
"""

import os
import glob
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def prep_esd(in_dir, out_dir):
    """
    ESD Structure:
    Speaker_0011/
       ├── Angry/
       │    ├── 0011_000351.wav
       ├── Neutral/
       ├── 0011.txt (Transcript: 0011_000351  "Text goes here")
    """
    print(f"Processing ESD from {in_dir}...")
    os.makedirs(out_dir, exist_ok=True)
    
    # ESD organizes by Speaker -> Emotion
    speakers = glob.glob(os.path.join(in_dir, "*"))
    
    for spk_path in speakers:
        if not os.path.isdir(spk_path): continue
        spk_id = os.path.basename(spk_path)
        
        # 1. Load Transcripts
        trans_map = {}
        txt_files = glob.glob(os.path.join(spk_path, "*.txt"))
        for txt in txt_files:
            with open(txt, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        # ESD format: FILENAME  TRANSCRIPT  EMOTION
                        fname = parts[0].strip()
                        text = parts[1].strip()
                        trans_map[fname] = text
        
        print(f"  Found {len(trans_map)} transcripts for {spk_id}")

        # 2. Copy Wavs and Create Txts
        wavs = glob.glob(os.path.join(spk_path, "**", "*.wav"), recursive=True)
        for wav_path in tqdm(wavs, desc=f"  Copying {spk_id}"):
            fname = os.path.splitext(os.path.basename(wav_path))[0]
            
            if fname not in trans_map:
                continue # Skip if no text found
                
            # Create standardized filename: 0011_Angry_000351.wav
            # (We preserve the emotion in the filename for future reference)
            parent_emotion = os.path.basename(os.path.dirname(wav_path))
            new_fname = f"{spk_id}_{parent_emotion}_{fname}.wav"
            
            # Copy Wav
            out_wav = os.path.join(out_dir, new_fname)
            shutil.copy2(wav_path, out_wav)
            
            # Write Text
            out_txt = os.path.join(out_dir, new_fname.replace('.wav', '.txt'))
            with open(out_txt, 'w', encoding='utf-8') as f:
                f.write(trans_map[fname])

def prep_jenny(in_dir, out_dir):
    """
    Jenny Structure:
    jenny/
      ├── metadata.csv (id|text|normalized_text)
      ├── wavs/
          ├── file1.wav
    """
    print(f"Processing Jenny from {in_dir}...")
    os.makedirs(out_dir, exist_ok=True)
    
    # Find metadata
    meta_path = os.path.join(in_dir, "metadata.csv")
    if not os.path.exists(meta_path):
        print("❌ metadata.csv not found in Jenny folder.")
        return

    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing Jenny"):
            parts = line.strip().split('|')
            if len(parts) < 2: continue
            
            fid = parts[0]
            text = parts[1]
            
            # Search for the wav file (Jenny structure varies by download source)
            # Try flat, or recursive
            wav_candidates = glob.glob(os.path.join(in_dir, "**", f"{fid}.wav"), recursive=True)
            if not wav_candidates:
                wav_candidates = glob.glob(os.path.join(in_dir, "**", f"{fid}.flac"), recursive=True)
                
            if wav_candidates:
                # Copy audio
                src_audio = wav_candidates[0]
                ext = os.path.splitext(src_audio)[1]
                dst_audio = os.path.join(out_dir, f"jenny_{fid}{ext}")
                
                # If flac, you might want to convert to wav here, but shutil copy is safer for now
                # We will just assume your loader handles FLAC via SoundFile/Librosa (it does)
                shutil.copy2(src_audio, dst_audio)
                
                # Write text
                with open(dst_audio.replace(ext, '.txt'), 'w', encoding='utf-8') as tf:
                    tf.write(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['esd', 'jenny'], required=True)
    parser.add_argument('--in_dir', required=True, help="Path to the downloaded raw dataset")
    parser.add_argument('--out_dir', required=True, help="Where to put the ready-to-train files")
    args = parser.parse_args()
    
    if args.dataset == 'esd':
        prep_esd(args.in_dir, args.out_dir)
    elif args.dataset == 'jenny':
        prep_jenny(args.in_dir, args.out_dir)
        
    print(f"\n✅ Done! Data is ready in {args.out_dir}")
    print(f"   Now run: python spev_real_metrics.py --mode train --data_dir {args.out_dir}")