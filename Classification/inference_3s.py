import torch
import torch.nn.functional as F
import numpy as np
import librosa
import matplotlib.pyplot as plt
import json
import sys
import os
from model import GenreCNN

# --- Config ---
MODEL_PATH = 'checkpoints/genre_cnn_pretrained_3schunks.pth'
GENRE_IDX_PATH = 'Data/data_preprocessed/genre_to_idx.json'
CHUNK_DURATION = 3  # seconds
N_MELS = 128
SR = 22050
CHUNK_TARGET_SHAPE = (1, 128, 129)

# --- Preprocessing (same as training) ---
def preprocess_audio_chunks(file_path, chunk_duration=3, n_mels=128, sr=22050):
    y, _ = librosa.load(file_path, sr=sr)
    total_duration = librosa.get_duration(y=y, sr=sr)
    chunks = []
    for start in np.arange(0, total_duration, chunk_duration):
        end = min(start + chunk_duration, total_duration)
        y_chunk = y[int(start * sr):int(end * sr)]
        if len(y_chunk) < chunk_duration * sr:
            y_chunk = np.pad(y_chunk, (0, int(chunk_duration * sr) - len(y_chunk)))
        mel = librosa.feature.melspectrogram(y=y_chunk, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
        mel_db_norm = mel_db_norm[np.newaxis, :, :]
        # Pad/crop to target shape
        c, h, w = mel_db_norm.shape
        tc, th, tw = CHUNK_TARGET_SHAPE
        if w < tw:
            pad_width = tw - w
            mel_db_norm = np.pad(mel_db_norm, ((0,0), (0,0), (0, pad_width)), mode='constant')
        elif w > tw:
            mel_db_norm = mel_db_norm[:, :, :tw]
        if h < th:
            pad_height = th - h
            mel_db_norm = np.pad(mel_db_norm, ((0,0), (0, pad_height), (0,0)), mode='constant')
        elif h > th:
            mel_db_norm = mel_db_norm[:, :th, :]
        chunks.append(mel_db_norm)
    return chunks

# --- Main inference function ---
def infer(file_path):
    # Load genre mapping
    with open(GENRE_IDX_PATH) as f:
        genre_to_idx = json.load(f)
    idx_to_genre = {v: k for k, v in genre_to_idx.items()}
    num_classes = len(genre_to_idx)

    # Load model
    model = GenreCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    # Preprocess audio into chunks
    chunks = preprocess_audio_chunks(file_path, chunk_duration=CHUNK_DURATION, n_mels=N_MELS, sr=SR)
    if not chunks:
        print("No valid audio chunks found.")
        return
    X = np.stack(chunks)
    X = torch.from_numpy(X).float()

    # Run inference
    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    avg_probs = probs.mean(axis=0)

    # Print top-2 genres
    top2_idx = avg_probs.argsort()[-2:][::-1]
    print("Top 2 genres:")
    for i in top2_idx:
        print(f"  {idx_to_genre[i]}: {avg_probs[i]*100:.2f}%")

    # Bar chart
    genres = [idx_to_genre[i] for i in range(num_classes)]
    plt.figure(figsize=(10, 5))
    plt.bar(genres, avg_probs)
    plt.ylabel('Probability')
    plt.title('Genre Probabilities')
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_img = os.path.splitext(os.path.basename(file_path))[0] + '_genre_probs.png'
    plt.savefig(out_img)
    print(f"Bar chart saved as {out_img}")
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inference_3s.py <audio_file.wav or .mp3>")
        sys.exit(1)
    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        sys.exit(1)
    infer(audio_path)
