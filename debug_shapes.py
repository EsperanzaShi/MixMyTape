#!/usr/bin/env python3
"""
Debug script to trace tensor shapes through the encoder.
"""

import torch
import numpy as np
from autoencoder import Autoencoder
from remix_mixtape import chunk_to_logmel, pad_or_crop_mel

def debug_encoder_shapes():
    """Debug the encoder to see where shape mismatch occurs."""
    
    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder(latent_dim=512, n_mels=80).to(device)
    checkpoint = torch.load('checkpoints/autoencoder_best.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Create test input
    sr = 22050
    duration = 3.0
    y = np.random.random(int(sr * duration))
    mel = chunk_to_logmel(y, sr=sr, n_mels=80, chunk_duration=3.0)
    mel = pad_or_crop_mel(mel, target_shape=(1, 80, 129))
    mel_tensor = torch.tensor(mel, dtype=torch.float32).to(device)
    
    print(f"Input tensor shape: {mel_tensor.shape}")
    print(f"Expected flattened size for FC layer: {model.encoder.fc.in_features}")
    
    # Trace through encoder manually
    x = mel_tensor
    print(f"\nStarting with: {x.shape}")
    
    # Conv1
    s1 = torch.relu(model.encoder.conv1(x))
    print(f"After conv1: {s1.shape}")
    
    # Conv2
    s2 = torch.relu(model.encoder.conv2(s1))
    print(f"After conv2: {s2.shape}")
    
    # Conv3
    s3 = torch.relu(model.encoder.conv3(s2))
    print(f"After conv3: {s3.shape}")
    
    # Conv4
    s4 = torch.relu(model.encoder.conv4(s3))
    print(f"After conv4: {s4.shape}")
    
    # Flatten using view
    z = s4.view(s4.size(0), -1)
    print(f"After flatten: {z.shape}")
    print(f"Expected shape: [batch, {256 * 80 * 9}]")
    
    # FC layer
    try:
        z = model.encoder.fc(z)
        print(f"After FC: {z.shape}")
    except Exception as e:
        print(f"Error in FC layer: {e}")
        print(f"FC layer expects: {model.encoder.fc.in_features} features")
        print(f"FC layer receives: {z.shape[1]} features")

if __name__ == '__main__':
    debug_encoder_shapes() 