import os
import librosa
import numpy as np
from tqdm import tqdm
import random
import json

def preprocess_audio(file_path, n_mels=128, duration=30, sr=22050):
    """
    Loads an audio file, converts it to a normalized Mel spectrogram.
    Args:
        file_path (str): Path to the .wav file.
        n_mels (int): Number of Mel bands to generate.
        duration (int): Duration to load (in seconds).
        sr (int): Sample rate.
    Returns:
        np.ndarray: Normalized Mel spectrogram (shape: [1, n_mels, time_steps]) or None if error
    """
    try:
        y, _ = librosa.load(file_path, sr=sr, duration=duration)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
        mel_db_norm = mel_db_norm[np.newaxis, :, :]
        return mel_db_norm
    except Exception as e:
        print(f"[Warning] Skipping {file_path}: {e}")
        return None


def process_dataset(dataset_dir, n_mels=128, duration=30, sr=22050):
    """
    Processes an entire dataset directory structured as genre folders containing .wav files.
    Returns a list of (mel_spectrogram, label) pairs.
    """
    data = []
    labels = []
    genres = sorted(os.listdir(dataset_dir))
    for genre in genres:
        genre_dir = os.path.join(dataset_dir, genre)
        if not os.path.isdir(genre_dir):
            continue
        for file in tqdm(os.listdir(genre_dir), desc=f"Processing {genre}"):
            if not file.endswith('.wav'):
                continue
            file_path = os.path.join(genre_dir, file)
            mel = preprocess_audio(file_path, n_mels=n_mels, duration=duration, sr=sr)
            if mel is not None:
                data.append(mel)
                labels.append(genre)
    return data, labels


def preprocess_audio_chunks(file_path, chunk_duration=3, n_mels=128, sr=22050):
    """
    Splits an audio file into chunks (default 3 seconds), returns a list of Mel spectrograms for each chunk.
    Skips the file if it cannot be loaded.
    """
    try:
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
            chunks.append(mel_db_norm)
        return chunks
    except Exception as e:
        print(f"[Warning] Skipping {file_path}: {e}")
        return []


def process_dataset_chunks(dataset_dir, chunk_duration=3, n_mels=128, sr=22050):
    """
    Processes an entire dataset directory, splitting each audio file into chunks (default 3 seconds).
    Returns a list of (mel_spectrogram_chunk, label) pairs.
    """
    data = []
    labels = []
    genres = sorted(os.listdir(dataset_dir))
    for genre in genres:
        genre_dir = os.path.join(dataset_dir, genre)
        if not os.path.isdir(genre_dir):
            continue
        for file in tqdm(os.listdir(genre_dir), desc=f"Chunking {genre}"):
            if not file.endswith('.wav'):
                continue
            file_path = os.path.join(genre_dir, file)
            chunks = preprocess_audio_chunks(file_path, chunk_duration=chunk_duration, n_mels=n_mels, sr=sr)
            data.extend(chunks)
            labels.extend([genre] * len(chunks))
    return data, labels


def stratified_split(dataset_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Stratified split of the dataset at the song level, ensuring balanced genre representation.
    Returns a dict with 'train', 'val', and 'test' keys, each containing a list of file paths.
    """
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

def is_mostly_silent(mel, threshold=0.01, min_nonzero_ratio=0.1):
    # mel: (1, n_mels, time)
    nonzero = np.sum(mel > threshold)
    total = mel.size
    return (nonzero / total) < min_nonzero_ratio

def save_preprocessed_split(splits, out_dir, n_mels=128, duration=30, sr=22050, chunk_duration=None, chunk_target_shape=(1,128,129)):
    os.makedirs(out_dir, exist_ok=True)
    for split_name, file_list in splits.items():
        split_dir = os.path.join(out_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for file_path in tqdm(file_list, desc=f"Processing {split_name} in {out_dir}"):
            base = os.path.splitext(os.path.basename(file_path))[0]
            if chunk_duration:
                # Save each chunk separately
                chunks = preprocess_audio_chunks(file_path, chunk_duration=chunk_duration, n_mels=n_mels, sr=sr)
                for i, chunk in enumerate(chunks):
                    if chunk is not None:
                        chunk = pad_or_crop(chunk, chunk_target_shape)
                        if not np.isnan(chunk).any() and not is_mostly_silent(chunk):
                            np.save(os.path.join(split_dir, f"{base}_chunk{i}.npy"), chunk)
                        else:
                            print(f"[Warning] Skipping silent or NaN chunk: {file_path} (chunk {i})")
            else:
                mel = preprocess_audio(file_path, n_mels=n_mels, duration=duration, sr=sr)
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
    save_preprocessed_split(splits, out_dir='Data/data_preprocessed/full_30s_n_mels80', n_mels=128, duration=30, sr=22050, chunk_duration=None)
    # Save 3s chunk spectrograms (target shape: 1,128,129)
    save_preprocessed_split(splits, out_dir='Data/data_preprocessed/chunks_3s_n_mels80', n_mels=128, duration=3, sr=22050, chunk_duration=3, chunk_target_shape=(1,128,129))
    # Save genre-to-index mapping
    genre_to_idx = get_genre_mapping('Data/genres_original')
    with open('Data/data_preprocessed/genre_to_idx_n_mels80.json', 'w') as f:
        json.dump(genre_to_idx, f)
    print("Genre to index mapping saved:", genre_to_idx)