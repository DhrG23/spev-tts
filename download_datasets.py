import os
import tarfile
import requests
import librosa
import pandas as pd
import soundfile as sf
from glob import glob
from pathlib import Path

# --- Configuration ---
# Multi-Speaker (LibriTTS-R)
MS_DATASET_URL = "http://www.openslr.org/resources/141/dev_clean.tar.gz"
MS_RAW_DIR = "data/raw_libri"
MS_OUTPUT_DIR = "data/training_data_libri"

# Single-Speaker (LJSpeech)
SS_DATASET_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
SS_RAW_DIR = "data/raw_ljspeech"
SS_OUTPUT_DIR = "data/training_data_ljspeech"

def download_and_extract(url, destination_dir, filename):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    archive_path = os.path.join(destination_dir, filename)
    
    if os.path.exists(archive_path) and os.path.getsize(archive_path) > 1000000:
        print(f"Archive {filename} already exists. Skipping download...")
    else:
        print(f"Downloading from {url}...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        with open(archive_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

    print(f"Extracting {filename}...")
    # Handle both .tar.gz and .tar.bz2
    mode = "r:gz" if filename.endswith("gz") else "r:bz2"
    with tarfile.open(archive_path, mode) as tar:
        tar.extractall(path=destination_dir)
    print("Extraction successful.")

def process_multi_speaker():
    """Processes LibriTTS-R full dataset"""
    os.makedirs(MS_OUTPUT_DIR, exist_ok=True)
    audio_files = glob(os.path.join(MS_RAW_DIR, "**/*.wav"), recursive=True)
    
    if not audio_files:
        print("No LibriTTS audio files found.")
        return

    print(f"Processing {len(audio_files)} Multi-Speaker files...")
    for count, audio_path in enumerate(audio_files, 1):
        txt_path = audio_path.replace(".wav", ".normalized.txt")
        if not os.path.exists(txt_path):
            txt_path = audio_path.replace(".wav", ".txt")
        
        if os.path.exists(txt_path):
            try:
                y, sr = librosa.load(audio_path, sr=22050)
                y_trimmed, _ = librosa.effects.trim(y, top_db=25)
                
                base_name = f"ms_sample_{count:05d}"
                sf.write(os.path.join(MS_OUTPUT_DIR, f"{base_name}.wav"), y_trimmed, 22050)
                
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                with open(os.path.join(MS_OUTPUT_DIR, f"{base_name}.txt"), 'w', encoding='utf-8') as f:
                    f.write(text)
            except Exception: continue
        if count % 500 == 0: print(f"Processed {count} files...")

def process_single_speaker():
    """Processes LJSpeech dataset"""
    os.makedirs(SS_OUTPUT_DIR, exist_ok=True)
    # LJSpeech extracts into a subfolder 'LJSpeech-1.1'
    base_path = Path(SS_RAW_DIR) / "LJSpeech-1.1"
    wav_dir = base_path / "wavs"
    metadata_path = base_path / "metadata.csv"

    if not metadata_path.exists():
        print("LJSpeech metadata not found!")
        return

    print("Processing Single-Speaker (LJSpeech) data...")
    # LJ001-0001 | Raw Text | Normalized Text
    df = pd.read_csv(metadata_path, sep='|', header=None, quoting=3)
    
    for i, row in df.iterrows():
        file_id = row[0]
        transcript = str(row[2]) # Use normalized column

        input_wav = wav_dir / f"{file_id}.wav"
        if input_wav.exists():
            try:
                # Standardize to 22050Hz, Mono for consistency
                y, sr = librosa.load(input_wav, sr=22050)
                y_trimmed, _ = librosa.effects.trim(y, top_db=25)
                y_norm = librosa.util.normalize(y_trimmed)

                sf.write(os.path.join(SS_OUTPUT_DIR, f"{file_id}.wav"), y_norm, 22050)
                with open(os.path.join(SS_OUTPUT_DIR, f"{file_id}.txt"), 'w', encoding='utf-8') as f:
                    f.write(transcript)
            except Exception: continue
        
        if (i + 1) % 500 == 0:
            print(f"Processed {i+1}/{len(df)} files...")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TTS Dataset Downloader and Preprocessor")
    parser.add_argument("--dataset", choices=["multi-speaker", "single-speaker", "both"], 
                        default="single-speaker", help="Which dataset to process")

    args = parser.parse_args()

    if args.dataset in ["multi-speaker", "both"]:
        print('='*60 + '\n' + ' '*7 +'Processing Multi-Speaker Dataset (LibriTTS-R)' + '\n' + '='*60)
        download_and_extract(MS_DATASET_URL, MS_RAW_DIR, "data.tar.gz")
        process_multi_speaker()
        
    if args.dataset in ["single-speaker", "both"]:
        print('='*60 + '\n' + ' '*8 +'Processing Single-Speaker Dataset (LJSpeech)' + '\n' + '='*60)
        download_and_extract(SS_DATASET_URL, SS_RAW_DIR, "ljspeech.tar.bz2")
        process_single_speaker()