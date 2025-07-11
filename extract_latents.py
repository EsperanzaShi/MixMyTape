import os
import glob
import numpy as np
import torch
from autoencoder import Autoencoder
import json
import librosa

# --- Config ---
AUDIO_DIR = 'chunked_wavs'
GENRES = ['blues', 'classical', 'jazz']
CKPT_PATH = 'checkpoints/autoencoder_best.pth'
LATENT_DIM = 512
N_MELS = 80
SR = 22050
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load model ---
model = Autoencoder(latent_dim=LATENT_DIM, n_mels=N_MELS).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()

latents = []
metadata = []

for genre in GENRES:
    genre_dir = os.path.join(AUDIO_DIR, genre)
    if not os.path.isdir(genre_dir):
        continue
    wav_files = glob.glob(os.path.join(genre_dir, '*.wav'))
    for wav_path in wav_files:
        # Load audio and convert to Mel (same as chunk_to_logmel)
        y, _ = librosa.load(wav_path, sr=SR, mono=True)
        mel = librosa.feature.melspectrogram(
            y=y, sr=SR, n_mels=N_MELS, n_fft=1024, hop_length=256, win_length=1024, fmin=0, fmax=8000, power=1.0
        )
        mel_log = np.log(np.clip(mel, a_min=1e-5, a_max=None))
        mel_log = np.clip(mel_log, a_min=-11.5129, a_max=2.0)
        mel_log = mel_log[np.newaxis, :, :]
        # Pad/crop to (1, 80, 129)
        c, h, w = mel_log.shape
        if w < 129:
            mel_log = np.pad(mel_log, ((0,0), (0,0), (0, 129-w)), mode='constant')
        elif w > 129:
            mel_log = mel_log[:, :, :129]
        mel_tensor = torch.tensor(mel_log, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, 1, 80, 129)
        with torch.no_grad():
            z, _ = model.encoder(mel_tensor)
        latents.append(z.cpu().numpy().flatten())
        metadata.append({'genre': genre, 'path': wav_path})
        print(f"Encoded: {wav_path}")

if len(latents) == 0:
    print("No latents found! Check your data paths and genre filters.")
else:
    latents = np.stack(latents)
    np.save('latents_blues_classical_jazz.npy', latents)
    with open('latents_blues_classical_jazz_meta.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {len(latents)} latents and metadata.") 