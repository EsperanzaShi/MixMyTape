import numpy as np
import librosa

# Pitch shift (expects waveform, not spectrogram)
def pitch_shift_waveform(y, sr, n_steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

# Time stretch (expects waveform, not spectrogram)
def time_stretch_waveform(y, rate):
    return librosa.effects.time_stretch(y, rate)

# Additive Gaussian noise (can be used on spectrograms or waveforms)
def add_noise(x, noise_level=0.01):
    noise = np.random.randn(*x.shape) * noise_level
    return x + noise

# Random gain (volume change)
def random_gain(x, min_gain=0.8, max_gain=1.2):
    gain = np.random.uniform(min_gain, max_gain)
    return x * gain

# Frequency masking (SpecAugment)
def freq_mask(x, F=15):
    x = x.copy()
    num_mel_channels = x.shape[1]
    f = np.random.randint(0, F)
    f0 = np.random.randint(0, num_mel_channels - f)
    x[:, f0:f0+f, :] = 0
    return x

# Time masking (SpecAugment)
def time_mask(x, T=20):
    x = x.copy()
    num_frames = x.shape[2]
    t = np.random.randint(0, T)
    t0 = np.random.randint(0, num_frames - t)
    x[:, :, t0:t0+t] = 0
    return x

# Example: Compose augmentations for a spectrogram (numpy array)
def augment_spectrogram(x, noise_level=0.01):
    x_aug = add_noise(x, noise_level=noise_level)
    return x_aug

# Aggressive/varied spectrogram augmentation
def aggressive_augment_spectrogram(x):
    if np.random.rand() < 0.5:
        x = add_noise(x, noise_level=0.05)
    if np.random.rand() < 0.5:
        x = random_gain(x)
    if np.random.rand() < 0.5:
        x = freq_mask(x, F=20)
    if np.random.rand() < 0.5:
        x = time_mask(x, T=30)
    return x

# Example: Compose augmentations for a waveform (numpy array)
def augment_waveform(y, sr, pitch_steps=2, stretch_rate=1.1, noise_level=0.01):
    y_aug = pitch_shift_waveform(y, sr, n_steps=np.random.uniform(-pitch_steps, pitch_steps))
    y_aug = time_stretch_waveform(y_aug, rate=np.random.uniform(1.0, stretch_rate))
    y_aug = add_noise(y_aug, noise_level=noise_level)
    return y_aug


