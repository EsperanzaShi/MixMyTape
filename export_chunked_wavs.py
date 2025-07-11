import os
import glob
import json
import soundfile as sf
import librosa
import numpy as np

# --- Config ---
META_PATH = 'latents_blues_classical_jazz_meta.json'
ORIG_AUDIO_DIR = 'Data/genres_original'
OUTPUT_DIR = 'chunked_wavs'
CHUNK_DURATION = 3.0  # seconds
SR = 22050

# --- Load metadata ---
with open(META_PATH) as f:
    meta = json.load(f)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for entry in meta:
    genre = entry['genre']
    npy_path = entry['path']
    fname = os.path.basename(npy_path)
    # Parse original wav basename and chunk index
    base, chunk_part = fname.split('_chunk')
    chunk_idx = int(chunk_part.split('.')[0])
    wav_name = base + '.wav'
    orig_wav_path = os.path.join(ORIG_AUDIO_DIR, genre, wav_name)
    if not os.path.isfile(orig_wav_path):
        print(f"Missing original wav: {orig_wav_path}")
        continue
    # Load original wav
    y, sr = librosa.load(orig_wav_path, sr=SR, mono=True)
    start = int(chunk_idx * CHUNK_DURATION * sr)
    end = int((chunk_idx + 1) * CHUNK_DURATION * sr)
    chunk = y[start:end]
    # Pad if needed
    if len(chunk) < int(CHUNK_DURATION * sr):
        chunk = np.pad(chunk, (0, int(CHUNK_DURATION * sr) - len(chunk)))
    # Output path
    out_dir = os.path.join(OUTPUT_DIR, genre)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base}_chunk{chunk_idx}.wav")
    sf.write(out_path, chunk, sr)
    print(f"Saved: {out_path}")

print("Done exporting all chunked wavs.") 