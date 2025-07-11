#!/usr/bin/env python3
"""
Test script to verify the remix pipeline works with correct Mel spectrogram shapes.
"""

import numpy as np
import torch
import soundfile as sf
from remix_mixtape import chunk_to_logmel, pad_or_crop_mel
from autoencoder import Autoencoder

def test_mel_consistency():
    """Test that Mel spectrograms are consistent between training and inference."""
    print("=== Testing Mel Spectrogram Consistency ===")
    
    # Generate 3 seconds of random audio
    sr = 22050
    duration = 3.0
    y = np.random.random(int(sr * duration))
    
    # Create Mel spectrogram using remix_mixtape logic
    mel = chunk_to_logmel(y, sr=sr, n_mels=80, chunk_duration=3.0)
    print(f"Original Mel shape: {mel.shape}")
    
    # Pad/crop to match training data
    mel_padded = pad_or_crop_mel(mel, target_shape=(1, 80, 129))
    print(f"Padded Mel shape: {mel_padded.shape}")
    
    # Load training data for comparison
    train_mel = np.load('Data/data_preprocessed/chunks_3s_n_mels80/train/rock.00049_chunk0.npy')
    print(f"Training data shape: {train_mel.shape}")
    
    # Verify shapes match
    assert mel_padded.shape == train_mel.shape, f"Shape mismatch: {mel_padded.shape} vs {train_mel.shape}"
    print("✅ Mel spectrogram shapes are consistent!")

def test_model_compatibility():
    """Test that the model can process the Mel spectrograms."""
    print("\n=== Testing Model Compatibility ===")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder(latent_dim=512, n_mels=80).to(device)
    model.eval()
    
    # Create test Mel spectrogram
    sr = 22050
    duration = 3.0
    y = np.random.random(int(sr * duration))
    mel = chunk_to_logmel(y, sr=sr, n_mels=80, chunk_duration=3.0)
    mel = pad_or_crop_mel(mel, target_shape=(1, 80, 129))
    
    # Convert to tensor
    mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 1, 80, 129)
    print(f"Input tensor shape: {mel_tensor.shape}")
    
    # Test encoder
    with torch.no_grad():
        z, skips = model.encoder(mel_tensor)
        print(f"Latent shape: {z.shape}")
        print(f"Number of skip connections: {len(skips)}")
        
        # Test full autoencoder
        recon = model(mel_tensor)
        print(f"Reconstruction shape: {recon.shape}")
    
    print("✅ Model can process Mel spectrograms successfully!")

def test_full_pipeline():
    """Test the complete remix pipeline with dummy data and skip connections."""
    print("\n=== Testing Full Pipeline ===")
    
    # Create dummy audio data
    sr = 22050
    duration = 3.0
    y1 = np.random.random(int(sr * duration))
    y2 = np.random.random(int(sr * duration))
    
    # Create Mel spectrograms
    mel1 = chunk_to_logmel(y1, sr=sr, n_mels=80, chunk_duration=3.0)
    mel1 = pad_or_crop_mel(mel1, target_shape=(1, 80, 129))
    mel2 = chunk_to_logmel(y2, sr=sr, n_mels=80, chunk_duration=3.0)
    mel2 = pad_or_crop_mel(mel2, target_shape=(1, 80, 129))
    
    # Convert to tensors with batch and channel dims
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mel_tensor1 = torch.tensor(mel1, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 1, 80, 129)
    mel_tensor2 = torch.tensor(mel2, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Load model
    model = Autoencoder(latent_dim=512, n_mels=80).to(device)
    checkpoint = torch.load('checkpoints/autoencoder_best.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Encode to (z, skips)
    with torch.no_grad():
        z1, skips1 = model.encoder(mel_tensor1)
        z2, skips2 = model.encoder(mel_tensor2)
    latents1 = [(z1, skips1)]
    latents2 = [(z2, skips2)]
    
    # Test remix_latents for all modes
    from remix_mixtape import remix_latents
    for mode in ['interleave', 'average', 'crossfade']:
        remixed = remix_latents(latents1, latents2, mode=mode)
        print(f"Testing remix mode: {mode}")
        for z, skips in remixed:
            with torch.no_grad():
                mel_recon = model.decoder(z, skips)
                assert mel_recon.shape == (1, 1, 80, 129), f"Reconstruction shape mismatch: {mel_recon.shape}"
        print(f"✅ {mode} remix mode works and produces correct shape.")
    print("Full encode-remix-decode pipeline with skips works!")

if __name__ == '__main__':
    test_mel_consistency()
    test_model_compatibility()
    test_full_pipeline()
    print("\n🎉 All tests passed! The remix pipeline should work correctly.") 