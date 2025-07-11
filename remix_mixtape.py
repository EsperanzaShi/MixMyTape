import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'hifi-gan'))
import numpy as np
import torch
from audio_utils import process_pair
import soundfile as sf
import librosa
from autoencoder import Autoencoder
from models import Generator  # HiFi-GAN
import json
from types import SimpleNamespace
import scipy.signal

# --- Config ---
SR = 22050
N_MELS = 80
LATENT_DIM = 512
CHUNK_DURATION = 3.0
MAX_CHUNKS = 10
AUTOENCODER_CKPT = 'checkpoints/autoencoder_best.pth'
HIFIGAN_CKPT = 'hifi-gan/UNIVERSAL_V1/g_02500000'
HIFIGAN_CONFIG = 'hifi-gan/UNIVERSAL_V1/config.json'

# Log-mel conversion (from waveform, not file)
def chunk_to_logmel(y, sr=SR, n_mels=N_MELS, chunk_duration=CHUNK_DURATION):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=256, win_length=1024, fmin=0, fmax=8000, power=1.0
    )
    mel_log = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    mel_log = np.clip(mel_log, a_min=-11.5129, a_max=2.0)
    mel_log = mel_log[np.newaxis, :, :]
    return mel_log

def is_percussive(y, sr, onset_thresh=0.3):
    """Return True if chunk is percussive based on onset strength."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    mean_onset = np.mean(onset_env)
    return mean_onset > onset_thresh

def remix_latents(latents1, latents2, mode='interleave', transition_chunks=3, chunks1_raw=None, chunks2_raw=None, sr=SR):
    remixed = []
    n = min(len(latents1), len(latents2))
    # If raw chunks are provided, use percussive detection
    percussive_mask = [False] * n
    if chunks1_raw is not None:
        percussive_mask = [is_percussive(y, sr) for y in chunks1_raw]
    if mode == 'interleave':
        for i in range(n):
            remixed.append(latents1[i] if i % 2 == 0 else latents2[i])
    elif mode == 'average':
        for i in range(n):
            z1, skips1 = latents1[i]
            z2, skips2 = latents2[i]
            z_avg = (z1 + z2) / 2
            skips_avg = []
            for s1, s2 in zip(skips1, skips2):
                if s1 is not None and s2 is not None:
                    skips_avg.append((s1 + s2) / 2)
                else:
                    skips_avg.append(s1 if s1 is not None else s2)
            remixed.append((z_avg, skips_avg))
    elif mode == 'weighted_blend':
        for i in range(n):
            if i < n // 2:
                weight = 0.8
                z1, skips1 = latents1[i]
                z2, skips2 = latents2[i]
                z_blend = weight * z1 + (1 - weight) * z2
                skips_blend = []
                for s1, s2 in zip(skips1, skips2):
                    if s1 is not None and s2 is not None:
                        skips_blend.append(weight * s1 + (1 - weight) * s2)
                    else:
                        skips_blend.append(s1 if s1 is not None else s2)
                remixed.append((z_blend, skips_blend))
            else:
                weight = 0.7
                z1, skips1 = latents1[i]
                z2, skips2 = latents2[i]
                z_blend = (1 - weight) * z1 + weight * z2
                skips_blend = []
                for s1, s2 in zip(skips1, skips2):
                    if s1 is not None and s2 is not None:
                        skips_blend.append((1 - weight) * s1 + weight * s2)
                    else:
                        skips_blend.append(s1 if s1 is not None else s2)
                remixed.append((z_blend, skips_blend))
    elif mode == 'crossfade':
        for i in range(n):
            if percussive_mask[i]:
                # For percussive, use only song 1
                remixed.append(latents1[i])
            else:
                alpha = i / max(n - 1, 1)
                z1, skips1 = latents1[i]
                z2, skips2 = latents2[i]
                z_mix = (1 - alpha) * z1 + alpha * z2
                remixed.append((z_mix, skips1))
    elif mode == 'transition_crossfade':
        # Use song 1, then crossfade, then song 2
        if transition_chunks >= n:
            transition_chunks = max(1, n // 3)
        pre = (n - transition_chunks) // 2
        post = n - transition_chunks - pre
        # First part: song 1
        for i in range(pre):
            remixed.append(latents1[i])
        # Crossfade part
        for i in range(transition_chunks):
            alpha = i / max(transition_chunks - 1, 1)
            z1, skips1 = latents1[pre + i]
            z2, skips2 = latents2[pre + i]
            z_mix = (1 - alpha) * z1 + alpha * z2
            skips_mix = []
            for s1, s2 in zip(skips1, skips2):
                if s1 is not None and s2 is not None:
                    skips_mix.append((1 - alpha) * s1 + alpha * s2)
                else:
                    skips_mix.append(s1 if s1 is not None else s2)
            remixed.append((z_mix, skips_mix))
        # Last part: song 2
        for i in range(post):
            remixed.append(latents2[pre + transition_chunks + i])
    else:
        raise ValueError(f'Unknown remix mode: {mode}')
    return remixed

# Pad or crop Mel spectrogram to (1, 80, 129) - EXACTLY like preprocessing
def pad_or_crop_mel(mel, target_shape=(1, 80, 129)):
    c, h, w = mel.shape
    tc, th, tw = target_shape
    # Pad or crop width (time axis)
    if w < tw:
        pad_width = tw - w
        mel = np.pad(mel, ((0,0), (0,0), (0, pad_width)), mode='constant')
    elif w > tw:
        mel = mel[:, :, :tw]
    # Pad or crop height (mel bins) if needed
    if h < th:
        pad_height = th - h
        mel = np.pad(mel, ((0,0), (0, pad_height), (0,0)), mode='constant')
    elif h > th:
        mel = mel[:, :th, :]
    return mel

def crossfade_audio(audio1, audio2, crossfade_samples=0, alpha=1.0):  # 0s at 22050Hz
    """Crossfade between two audio chunks.
    
    Args:
        audio1: First audio chunk
        audio2: Second audio chunk  
        crossfade_samples: Number of samples to crossfade
        alpha: Crossfade curve shape (1.0=linear, 2.0=exponential, 0.5=square root)
    """
    if len(audio1) < crossfade_samples or len(audio2) < crossfade_samples:
        return np.concatenate([audio1, audio2])
    
    # Create fade curves with adjustable shape
    t = np.linspace(0, 1, crossfade_samples)
    fade_out = (1 - t**alpha)  # Fade out curve
    fade_in = t**alpha         # Fade in curve
    
    # Apply crossfade - fix the indexing
    audio1_fade = audio1[-crossfade_samples:] * fade_out  # Last crossfade_samples of audio1
    audio2_fade = audio2[:crossfade_samples] * fade_in    # First crossfade_samples of audio2
    
    # Combine
    crossfaded = audio1_fade + audio2_fade
    remaining = audio2[crossfade_samples:]
    
    return np.concatenate([audio1[:-crossfade_samples], crossfaded, remaining])

def smooth_concatenate(audio_chunks, crossfade_samples=2205, alpha=1.0):
    """Smoothly concatenate audio chunks with crossfading.
    
    Args:
        audio_chunks: List of audio chunks
        crossfade_samples: Number of samples to crossfade
        alpha: Crossfade curve shape (1.0=linear, 2.0=exponential, 0.5=square root)
    """
    if len(audio_chunks) == 0:
        return np.array([])
    elif len(audio_chunks) == 1:
        return audio_chunks[0]
    
    result = audio_chunks[0]
    for chunk in audio_chunks[1:]:
        result = crossfade_audio(result, chunk, crossfade_samples, alpha)
    
    return result

def apply_eq(audio, sr=SR, lowcut=180, highcut=600, gain_db=-6):
    """Apply a simple EQ to reduce muddy frequencies (200-500 Hz)."""
    # Design a peaking filter
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.iirfilter(2, [low, high], btype='band', ftype='butter')
    filtered = scipy.signal.lfilter(b, a, audio)
    # Mix dry/wet for gentle effect
    gain = 10 ** (gain_db / 20)
    return audio + gain * (filtered - audio)

def main(wav1, wav2, output_wav='mixtape.wav', remix_mode='interleave', crossfade_duration=0.1, alpha=1.0):
    # 1. Chunk and time-stretch
    print('Chunking and time-stretching...')
    chunks1, chunks2, bpm1, bpm2, target_bpm = process_pair(wav1, wav2, sr=SR, chunk_duration=CHUNK_DURATION, max_chunks=MAX_CHUNKS)
    print(f'Processing {len(chunks1)} chunks per song at BPM {target_bpm:.2f}')
    print(f'Remix mode: {remix_mode}')
    print(f'Crossfade duration: {crossfade_duration:.2f}s')
    print(f'Crossfade alpha: {alpha:.2f}')

    # 2. Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder = Autoencoder(latent_dim=LATENT_DIM, n_mels=N_MELS).to(device)
    autoencoder.load_state_dict(torch.load(AUTOENCODER_CKPT, map_location=device))
    autoencoder.eval()
    with open(HIFIGAN_CONFIG) as f:
        h = json.load(f)
    h = SimpleNamespace(**h)
    generator = Generator(h).to(device)
    state_dict_g = torch.load(HIFIGAN_CKPT, map_location=device)
    if "generator" in state_dict_g:
        state_dict_g = state_dict_g["generator"]
    generator.load_state_dict(state_dict_g)
    generator.eval()

    # 3. Encode chunks to latents
    print('Encoding chunks to latents...')
    latents1, latents2 = [], []
    with torch.no_grad():
        for chunk in chunks1:
            mel = chunk_to_logmel(chunk, sr=SR, n_mels=N_MELS, chunk_duration=CHUNK_DURATION)
            mel = pad_or_crop_mel(mel, target_shape=(1, N_MELS, 129))
            mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 1, 80, 129)
            z, skips = autoencoder.encoder(mel_tensor)
            latents1.append((z, skips))
        for chunk in chunks2:
            mel = chunk_to_logmel(chunk, sr=SR, n_mels=N_MELS, chunk_duration=CHUNK_DURATION)
            mel = pad_or_crop_mel(mel, target_shape=(1, N_MELS, 129))
            mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 1, 80, 129)
            z, skips = autoencoder.encoder(mel_tensor)
            latents2.append((z, skips))

    # 4. Remix latents
    print('Remixing latents...')
    remixed_latents = remix_latents(latents1, latents2, mode=remix_mode, chunks1_raw=chunks1, sr=SR)

    # 5. Decode latents to Mel, then to audio
    print('Decoding and vocoding...')
    mixtape_audio = []
    with torch.no_grad():
        for z, skips in remixed_latents:
            mel_recon = autoencoder.decoder(z, skips)
            mel_recon = mel_recon.cpu().numpy().squeeze()  # Remove batch and channel dims: (1, 1, 80, 129) -> (80, 129)
            mel_tensor = torch.tensor(mel_recon, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 80, 129) for HiFi-GAN
            audio = generator(mel_tensor).cpu().numpy().squeeze()
            mixtape_audio.append(audio)

    # 6. Concatenate audio with crossfading
    print('Concatenating audio chunks...')
    crossfade_samples = int(crossfade_duration * SR)
    mixtape = smooth_concatenate(mixtape_audio, crossfade_samples=crossfade_samples, alpha=alpha)
    # Apply EQ to reduce muddiness
    mixtape = apply_eq(mixtape, sr=SR, lowcut=180, highcut=600, gain_db=-6)
    sf.write(output_wav, mixtape, SR)
    print(f'Mixtape saved to {output_wav}')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3 or len(sys.argv) > 8:
        print('Usage: python remix_mixtape.py file1.wav file2.wav [remix_mode] [crossfade_duration] [alpha] [transition_chunks] [output_file]')
        print('remix_mode: interleave | average | weighted_blend | crossfade | transition_crossfade (default: interleave)')
        print('crossfade_duration: seconds (default: 0.0)')
        print('alpha: crossfade curve shape (default: 1.0)')
        print('  - 1.0: linear crossfade')
        print('  - 2.0: exponential (more aggressive)')
        print('  - 0.5: square root (gentler)')
        print('transition_chunks: number of chunks to crossfade in transition_crossfade mode (default: 3, only used for transition_crossfade)')
        print('output_file: output filename (default: mixtape.wav)')
        exit(1)
    wav1, wav2 = sys.argv[1], sys.argv[2]
    remix_mode = sys.argv[3] if len(sys.argv) > 3 else 'interleave'
    crossfade_duration = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    alpha = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    transition_chunks = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    output_file = sys.argv[7] if len(sys.argv) > 7 else 'mixtape.wav'
    # Patch remix_latents only if needed
    if remix_mode == 'transition_crossfade':
        def remix_latents_with_transition(*args, **kwargs):
            return remix_latents(*args, mode=remix_mode, transition_chunks=transition_chunks)
        import builtins
        builtins.remix_latents = remix_latents_with_transition
        main(wav1, wav2, output_file, remix_mode, crossfade_duration, alpha)
        builtins.remix_latents = remix_latents
    else:
        main(wav1, wav2, output_file, remix_mode, crossfade_duration, alpha) 