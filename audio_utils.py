import librosa
import numpy as np
import soundfile as sf

def detect_bpm(audio_path, sr=22050):
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)

def time_stretch_to_bpm(y, orig_bpm, target_bpm, sr=22050):
    # Ensure y is 1D (mono)
    if y.ndim > 1:
        y = librosa.to_mono(y)
    rate = target_bpm / orig_bpm
    # Compute STFT
    D = librosa.stft(y)
    # Time-stretch the STFT
    D_stretch = librosa.phase_vocoder(D, rate=rate)
    # Inverse STFT to get back to waveform
    y_stretched = librosa.istft(D_stretch, length=int(len(y) / rate))
    return y_stretched

def chunk_audio(y, sr=22050, chunk_duration=3.0, max_chunks=None, overlap=0.5):
    """
    Chunk audio with optional overlap for smoother transitions.
    
    Args:
        y: Audio waveform
        sr: Sample rate
        chunk_duration: Duration of each chunk in seconds
        max_chunks: Maximum number of chunks to create
        overlap: Overlap between chunks (0.0 = no overlap, 1.0 = full overlap)
    """
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(chunk_samples * overlap)
    step_samples = chunk_samples - overlap_samples
    
    total_samples = len(y)
    chunks = []
    
    for i, start in enumerate(range(0, total_samples, step_samples)):
        if max_chunks is not None and i >= max_chunks:
            break
        end = min(start + chunk_samples, total_samples)
        chunk = y[start:end]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append(chunk)
    
    return chunks

def process_pair(wav1, wav2, sr=22050, chunk_duration=3.0, max_chunks=10):
    y1, _ = librosa.load(wav1, sr=sr, mono=True)
    y2, _ = librosa.load(wav2, sr=sr, mono=True)
    bpm1 = detect_bpm(wav1, sr)
    bpm2 = detect_bpm(wav2, sr)
    print(f'BPM1: {bpm1:.2f}, BPM2: {bpm2:.2f}')
    if bpm1 >= bpm2:
        target_bpm = bpm1
        y2_stretched = time_stretch_to_bpm(y2, bpm2, target_bpm, sr)
        y1_stretched = y1
    else:
        target_bpm = bpm2
        y1_stretched = time_stretch_to_bpm(y1, bpm1, target_bpm, sr)
        y2_stretched = y2
    chunks1 = chunk_audio(y1_stretched, sr, chunk_duration, max_chunks)
    chunks2 = chunk_audio(y2_stretched, sr, chunk_duration, max_chunks)
    return chunks1, chunks2, bpm1, bpm2, target_bpm

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print('Usage: python audio_utils.py file1.wav file2.wav')
        exit(1)
    wav1, wav2 = sys.argv[1], sys.argv[2]
    chunks1, chunks2, bpm1, bpm2, target_bpm = process_pair(wav1, wav2, max_chunks=10)
    print(f'Chunks for {wav1}: {len(chunks1)}')
    print(f'Chunks for {wav2}: {len(chunks2)}')
    # Example: save first chunk of each (comment out if not needed)
    # sf.write('chunk1_0.wav', chunks1[0], 22050)
    # sf.write('chunk2_0.wav', chunks2[0], 22050)
    print('First chunk of each file is available in memory (not saved by default).') 