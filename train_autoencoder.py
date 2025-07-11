import sys
sys.path.append('/Users/qs20/Desktop/ML institue/audioML/hifi-gan')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import wandb
import matplotlib.pyplot as plt
from autoencoder import Autoencoder
from Classification.dataset import SpectrogramDataset
from tqdm.auto import tqdm
from models import Generator  # from hifi-gan repo
import json
from types import SimpleNamespace
import librosa
import scipy.signal

# --- Config ---
BATCH_SIZE = 128 #32
EPOCHS = 40 #20
FREEZE_EPOCHS = 0 #20
LATENT_DIM = 512 #128, 256
N_MELS = 80
LEARNING_RATE = 1e-3
CHECKPOINT_PATH = 'checkpoints/autoencoder_best.pth'
#ENCODER_CKPT = 'checkpoints/genre_cnn_medium.pth'
TRAIN_DIR = 'Data/data_preprocessed/chunks_3s_n_mels80/train'
VAL_DIR = 'Data/data_preprocessed/chunks_3s_n_mels80/val'
GENRE_IDX_PATH = 'Data/data_preprocessed/genre_to_idx_n_mels80.json'
HIFIGAN_CKPT = '/Users/qs20/Desktop/ML institue/audioML/hifi-gan/UNIVERSAL_V1/g_02500000'
HIFIGAN_CONFIG = '/Users/qs20/Desktop/ML institue/audioML/hifi-gan/UNIVERSAL_V1/config.json'

# --- Utility: Save spectrogram as image ---
def save_spectrogram_image(mel, out_path):
    plt.figure(figsize=(6, 3))
    plt.imshow(mel.squeeze(), aspect='auto', origin='lower', cmap='magma')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# --- Main training function ---
def train_autoencoder():
    # Load genre mapping (for dataset)
    with open(GENRE_IDX_PATH) as f:
        genre_to_idx = json.load(f)

    # Datasets and loaders
    train_dataset = SpectrogramDataset(TRAIN_DIR, genre_to_idx)
    val_dataset = SpectrogramDataset(VAL_DIR, genre_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model
    model = Autoencoder(latent_dim=LATENT_DIM,n_mels=N_MELS)
    #model.load_encoder_weights(ENCODER_CKPT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load HiFi-GAN config and generator
    with open(HIFIGAN_CONFIG) as f:
        h = json.load(f)
    h = SimpleNamespace(**h)
    generator = Generator(h).to(device)
    state_dict_g = torch.load(HIFIGAN_CKPT, map_location=device)
    if "generator" in state_dict_g:
        state_dict_g = state_dict_g["generator"]
    generator.load_state_dict(state_dict_g)
    generator.eval()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # WandB
    wandb.init(project='audio-autoencoder', entity='week5', config={
        'latent_dim': LATENT_DIM,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'freeze_encoder_epochs': FREEZE_EPOCHS
    })

    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        # Optionally freeze encoder
        if epoch < FREEZE_EPOCHS:
            for param in model.encoder.parameters():
                param.requires_grad = False
        else:
            for param in model.encoder.parameters():
                param.requires_grad = True

        # Training
        model.train()
        train_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        for x, _ in train_iter:
            x = x.to(device)
            optimizer.zero_grad()
            x_recon = model(x)
            loss = criterion(x_recon, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_iter.set_postfix(loss=loss.item())
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for x, _ in val_iter:
                x = x.to(device)
                x_recon = model(x)
                loss = criterion(x_recon, x)
                val_loss += loss.item() * x.size(0)
                val_iter.set_postfix(loss=loss.item())
        val_loss /= len(val_loader.dataset)

        # Log to wandb
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)

            # Log a few reconstructions to wandb for visual inspection
            x_vis = next(iter(val_loader))[0][:3].to(device)
            model.eval()
            with torch.no_grad():
                x_recon_vis = model(x_vis)
            images = []
            audios = []
            sr = 22050  # Use your dataset's sample rate
            for i in range(3):
                fig, axes = plt.subplots(1, 2, figsize=(8, 3))
                axes[0].imshow(x_vis[i].cpu().numpy().squeeze(), aspect='auto', origin='lower', cmap='magma')
                axes[0].set_title('Input')
                axes[1].imshow(x_recon_vis[i].cpu().numpy().squeeze(), aspect='auto', origin='lower', cmap='magma')
                axes[1].set_title('Reconstruction')
                plt.tight_layout()
                images.append(wandb.Image(fig, caption=f"Sample {i}"))
                plt.close(fig)
                # Convert reconstructed Mel to audio (HiFi-GAN)
                mel = x_recon_vis[i].cpu().numpy().squeeze()  # (128, T)
                mel_80 = scipy.signal.resample(mel, 80, axis=0)  # Resample along the frequency axis
                mel_tensor = torch.from_numpy(mel_80).unsqueeze(0).to(device)  # (1, 80, T)
                with torch.no_grad():
                    audio = generator(mel_tensor).cpu().numpy().squeeze()
                audios.append(wandb.Audio(audio, sample_rate=sr, caption=f"Recon Sample {i} (HiFi-GAN)"))
            wandb.log({'reconstructions': images, 'recon_audio': audios})

    wandb.finish()

if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    train_autoencoder() 