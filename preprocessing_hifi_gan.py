# Preprocessing for Mel spectrograms compatible with HiFi-GAN UNIVERSAL_V1
# n_mels=80, n_fft=1024, hop_length=256, win_length=1024, fmin=0, fmax=8000, sr=22050

import os
import librosa
import numpy as np
from tqdm import tqdm
import random
import json

def preprocess_audio(file_path, n_mels=80, duration=30, sr=22050, n_fft=1024, hop_length=256, win_length=1024, fmin=0, fmax=8000):
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=duration)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length, fmin=fmin, fmax=fmax, power=1.0
        )
        mel_log = np.log(np.clip(mel, a_min=1e-5, a_max=None))
        mel_log = np.clip(mel_log, a_min=-11.5129, a_max=2.0)
        mel_log = mel_log[np.newaxis, :, :]
        return mel_log
    except Exception as e:
        print(f"[Warning] Skipping {file_path}: {e}")
        return None

def preprocess_audio_chunks(file_path, chunk_duration=3, n_mels=80, sr=22050, n_fft=1024, hop_length=256, win_length=1024, fmin=0, fmax=8000):
    try:
        y, _ = librosa.load(file_path, sr=sr)
        total_duration = librosa.get_duration(y=y, sr=sr)
        chunks = []
        for start in np.arange(0, total_duration, chunk_duration):
            end = min(start + chunk_duration, total_duration)
            y_chunk = y[int(start * sr):int(end * sr)]
            if len(y_chunk) < chunk_duration * sr:
                y_chunk = np.pad(y_chunk, (0, int(chunk_duration * sr) - len(y_chunk)))
            mel = librosa.feature.melspectrogram(
                y=y_chunk, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length, fmin=fmin, fmax=fmax, power=1.0
            )
            mel_log = np.log(np.clip(mel, a_min=1e-5, a_max=None))
            mel_log = np.clip(mel_log, a_min=-11.5129, a_max=2.0)
            mel_log = mel_log[np.newaxis, :, :]
            chunks.append(mel_log)
        return chunks
    except Exception as e:
        print(f"[Warning] Skipping {file_path}: {e}")
        return []

def stratified_split(dataset_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    random.seed(seed)
    splits = {'train': [], 'val': [], 'test': []}
    genres = sorted(os.listdir(dataset_dir))
    for genre in genres:
        genre_dir = os.path.join(dataset_dir, genre)
        if not os.path.isdir(genre_dir):
            continue
        files = [os.path.join(genre_dir, f) for f in os.listdir(genre_dir) if f.endswith('.wav')]
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits['train'].extend(files[:n_train])
        splits['val'].extend(files[n_train:n_train + n_val])
        splits['test'].extend(files[n_train + n_val:])
    return splits

def pad_or_crop(mel, target_shape):
    c, h, w = mel.shape
    tc, th, tw = target_shape
    # Pad or crop width (time axis)
    if w < tw:
        pad_width = tw - w
        mel = np.pad(mel, ((0,0), (0,0), (0, pad_width)), mode='constant')
    elif w > tw:
        mel = mel[:, :, :tw]
    # Optionally pad/crop height (mel bands) if needed
    if h < th:
        pad_height = th - h
        mel = np.pad(mel, ((0,0), (0, pad_height), (0,0)), mode='constant')
    elif h > th:
        mel = mel[:, :th, :]
    return mel

def is_mostly_silent(mel, threshold=-11, min_nonzero_ratio=0.1):
    nonzero = np.sum(mel > threshold)
    total = mel.size
    return (nonzero / total) < min_nonzero_ratio

def save_preprocessed_split(splits, out_dir, n_mels=80, duration=30, sr=22050, chunk_duration=None, chunk_target_shape=(1,80,129), n_fft=1024, hop_length=256, win_length=1024, fmin=0, fmax=8000):
    os.makedirs(out_dir, exist_ok=True)
    for split_name, file_list in splits.items():
        split_dir = os.path.join(out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for file_path in tqdm(file_list, desc=f"Processing {split_name} in {out_dir}"):
            base = os.path.splitext(os.path.basename(file_path))[0]
            if chunk_duration:
                chunks = preprocess_audio_chunks(file_path, chunk_duration=chunk_duration, n_mels=n_mels, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, fmin=fmin, fmax=fmax)
                for i, chunk in enumerate(chunks):
                    if chunk is not None:
                        chunk = pad_or_crop(chunk, chunk_target_shape)
                        if not np.isnan(chunk).any() and not is_mostly_silent(chunk):
                            np.save(os.path.join(split_dir, f"{base}_chunk{i}.npy"), chunk)
                        else:
                            print(f"[Warning] Skipping silent or NaN chunk: {file_path} (chunk {i})")
            else:
                mel = preprocess_audio(file_path, n_mels=n_mels, duration=duration, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, fmin=fmin, fmax=fmax)
                if mel is not None:
                    if not np.isnan(mel).any():
                        np.save(os.path.join(split_dir, f"{base}.npy"), mel)
                    else:
                        print(f"[Warning] Skipping file with NaN values: {file_path}")

def get_genre_mapping(dataset_dir):
    genres = sorted([g for g in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, g))])
    genre_to_idx = {genre: idx for idx, genre in enumerate(genres)}
    return genre_to_idx

if __name__ == '__main__':
    splits = stratified_split('Data/genres_original')
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    # Save full 30s spectrograms
    save_preprocessed_split(splits, out_dir='Data/data_preprocessed/full_30s_n_mels80', n_mels=80, duration=30, sr=22050, chunk_duration=None, chunk_target_shape=(1,80,129), n_fft=1024, hop_length=256, win_length=1024, fmin=0, fmax=8000)
    # Save 3s chunk spectrograms (target shape: 1,80,129)
    save_preprocessed_split(splits, out_dir='Data/data_preprocessed/chunks_3s_n_mels80', n_mels=80, duration=3, sr=22050, chunk_duration=3, chunk_target_shape=(1,80,129), n_fft=1024, hop_length=256, win_length=1024, fmin=0, fmax=8000)
    # Save genre-to-index mapping
    genre_to_idx = get_genre_mapping('Data/genres_original')
    with open('Data/data_preprocessed/genre_to_idx_n_mels80.json', 'w') as f:
        json.dump(genre_to_idx, f)
    print("Genre to index mapping saved:", genre_to_idx)
