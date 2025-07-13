import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import glob
import numpy as np
import time
from tqdm import tqdm
import argparse
import random

sr = 44100
n_mels = 128
n_fft = 2048
hop_length = 256
f_min=50
f_max=16000
duration=3.0

# Label map for IRMAS folders (updated to match IRMAS exactly)
LABEL_MAP = {
    "cel": 0,  # cello
    "cla": 1,  # clarinet
    "flu": 2,  # flute
    "gac": 3,  # acoustic guitar
    "gel": 4,  # electric guitar
    "org": 5,  # organ
    "pia": 6,  # piano
    "sax": 7,  # saxophone
    "tru": 8,  # trumpet
    "vio": 9,  # violin
    "voi": 10, # human singing voice
}

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def ensure_model_dir():
    if not os.path.exists("model"):
        os.makedirs("model")

def spectral_gate(mel_spec, threshold_db=-40):
    # Convert to dB
    mel_db = torchaudio.functional.amplitude_to_DB(mel_spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0, top_db=80.0)
    mask = mel_db > threshold_db
    return mel_spec * mask

# Precompute features and labels for a dataset split

def precompute_features(root_dirs, label_map, out_prefix, duration=duration, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, f_min=f_min, f_max=f_max):
    import itertools
    audio_exts = ["wav", "mp3", "flac", "aiff", "ogg"]
    samples = []
    labels = []
    # Accept a list of root_dirs
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]
    for root_dir in root_dirs:
        for folder_code, label_idx in label_map.items():
            folder = os.path.join(root_dir, folder_code)
            found = list(itertools.chain.from_iterable(
                glob.glob(os.path.join(folder, f"**/*.{ext}"), recursive=True) for ext in audio_exts
            ))
            samples.extend(found)
            labels.extend([label_idx] * len(found))
            print(f"Found {len(found)} files for {folder_code} in {folder}")
    if not samples:
        print(f"Warning: No audio files found in {root_dirs} for {out_prefix}. Skipping precompute.")
        return
    features = []
    new_labels = []
    for wav_path, label in tqdm(list(zip(samples, labels)), desc=f"Precomputing {out_prefix}"):
        waveform, orig_sr = torchaudio.load(wav_path)
        if orig_sr != sr:
            waveform = torchaudio.transforms.Resample(orig_sr, sr)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        num_samples = int(sr * duration)
        if waveform.shape[1] > num_samples:
            waveform = waveform[:, :num_samples]
        elif waveform.shape[1] < num_samples:
            pad = num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )(waveform)
        # Apply spectral gating
        mel_spec = spectral_gate(mel_spec, threshold_db=-40)
        # Add small epsilon before log to prevent NaN values
        # mel_spec = torch.log2(mel_spec + 1e-8)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
        features.append(mel_spec)
        new_labels.append(label)
    features = torch.stack(features)
    new_labels = torch.tensor(new_labels)
    ensure_model_dir()
    torch.save(features, f"model/{out_prefix}_features.pt")
    torch.save(new_labels, f"model/{out_prefix}_labels.pt")
    print(f"Saved {features.shape[0]} samples to model/{out_prefix}_features.pt and model/{out_prefix}_labels.pt")

class IRMASPrecomputedDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return self.features.shape[0]
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"batch_loss": loss.item()})
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    examples_printed = 0
    example_limit = 3
    with torch.no_grad():
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            out = model(X)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            # Print up to 3 (pred, true) examples
            if examples_printed < example_limit:
                for i in range(X.size(0)):
                    if examples_printed < example_limit:
                        pred_label = INV_LABEL_MAP[preds[i].item()]
                        true_label = INV_LABEL_MAP[y[i].item()]
                        print(f"Example {examples_printed+1}: Predicted={pred_label}, True={true_label}")
                        examples_printed += 1
    acc = correct / total if total > 0 else 0.0
    return acc

def select_device():
    """Select the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout"""
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        return out

class EfficientSelfAttention(nn.Module):
    """Efficient self-attention mechanism using spatial reduction"""
    def __init__(self, channels, num_heads=8, reduction_ratio=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert self.head_dim * num_heads == channels, "channels must be divisible by num_heads"
        self.reduction_ratio = reduction_ratio
        
        # Spatial reduction for efficiency
        self.sr = nn.Conv2d(channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio)
        self.norm = nn.LayerNorm(channels)
        
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Spatial reduction to reduce computational cost
        x_reduced = self.sr(x)  # (B, C, H//r, W//r)
        H_reduced, W_reduced = x_reduced.shape[2], x_reduced.shape[3]
        
        # Generate Q, K, V from reduced spatial dimensions
        q = self.query(x_reduced).view(B, self.num_heads, self.head_dim, H_reduced * W_reduced)
        k = self.key(x_reduced).view(B, self.num_heads, self.head_dim, H_reduced * W_reduced)
        v = self.value(x_reduced).view(B, self.num_heads, self.head_dim, H_reduced * W_reduced)
        
        # Compute attention (much smaller matrix now)
        attn = torch.softmax(torch.matmul(q.transpose(-2, -1), k) / (self.head_dim ** 0.5), dim=-1)
        out = torch.matmul(v, attn.transpose(-2, -1))
        
        # Reshape and project
        out = out.view(B, C, H_reduced, W_reduced)
        out = self.proj(out)
        
        # Upsample back to original spatial dimensions
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        
        # Residual connection and normalization
        out = out + x
        out = out.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
        out = self.norm(out)
        out = out.transpose(1, 2).view(B, C, H, W)  # (B, C, H, W)
        
        return out

class SEModule(nn.Module):
    """Squeeze-and-Excitation module for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Squeeze: global average and max pooling
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        
        # Excitation: learn channel-wise attention weights
        avg_weights = self.fc(avg_out)
        max_weights = self.fc(max_out)
        
        # Combine and apply attention
        attention_weights = (avg_weights + max_weights).view(b, c, 1, 1)
        return x * attention_weights

class AttentionModule(nn.Module):
    """Channel attention module to focus on important frequency bands"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        out = avg_out + max_out
        attention_weights = self.sigmoid(out).view(b, c, 1, 1)
        
        return x * attention_weights

class ImprovedCNN(nn.Module):
    """Improved CNN with residual connections, attention, and better feature extraction"""
    def __init__(self, n_classes, input_channels=1, dropout_rate=0.3):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks with increasing channels
        self.layer1 = self._make_layer(32, 64, 2, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, dropout_rate=dropout_rate)
        
        # Attention modules
        self.attention1 = AttentionModule(64)
        self.attention2 = AttentionModule(128)
        self.attention3 = AttentionModule(256)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier with multiple layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, n_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride, dropout_rate):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, dropout_rate))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers with attention
        x = self.layer1(x)
        x = self.attention1(x)
        
        x = self.layer2(x)
        x = self.attention2(x)
        
        x = self.layer3(x)
        x = self.attention3(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

class SimpleCNN(nn.Module):
    """Fast SimpleCNN with residual connections but no heavy attention"""
    def __init__(self, n_classes):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Residual blocks (keep the good parts!)
        # self.res_block1 = ResidualBlock(16, 32, stride=1)
        # self.res_block2 = ResidualBlock(32, 64, stride=1)
        # self.res_block3 = ResidualBlock(128, 256, stride=1)
        
        # Global pooling and classification
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.1)  # Reduced dropout
        self.fc = nn.Linear(128, n_classes)
        
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling and classification
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        return self.fc(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
    def forward(self, x):
        return self.layer(x)

class CrossAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
    def forward(self, x, x_cross):
        attn_out, _ = self.cross_attn(x, x_cross, x_cross)
        x = self.norm(x + attn_out)
        x = self.norm(x + self.ff(x))
        return x

class CrossDomainTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers=5, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout))
            if i < num_layers - 1:
                self.layers.append(CrossAttentionEncoderLayer(d_model, nhead, dropout))
    def forward(self, x_freq, x_time):
        for layer in self.layers:
            if isinstance(layer, TransformerEncoderLayer):
                x_freq = layer(x_freq)
            elif isinstance(layer, CrossAttentionEncoderLayer):
                x_freq = layer(x_freq, x_time)
        return x_freq

class HybridTransformerInstrumentClassifier(nn.Module):
    def __init__(self, n_classes, d_model=128, nhead=4, num_layers=5, n_mels=n_mels, sr=sr, duration=3.0):
        super().__init__()
        self.n_mels = n_mels
        self.sr = sr
        self.duration = duration
        self.samples = int(sr * duration)
        # Input projections for time and frequency domains
        self.freq_proj = nn.Linear(n_mels, d_model)  # (B, T, n_mels) -> (B, T, d_model)
        self.time_proj = nn.Linear(self.samples, d_model)  # (B, samples) -> (B, d_model)
        self.encoder = CrossDomainTransformerEncoder(d_model, nhead, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x_mel, x_wave):
        # x_mel: (B, 1, n_mels, T) or (B, n_mels, T)
        # x_wave: (B, samples)
        if x_mel.dim() == 4:
            x_mel = x_mel.squeeze(1)  # (B, n_mels, T)
        x_freq = x_mel.transpose(1, 2)  # (B, T, n_mels)
        x_freq = self.freq_proj(x_freq)  # (B, T, d_model)
        x_time = self.time_proj(x_wave).unsqueeze(1)  # (B, 1, d_model)
        x = self.encoder(x_freq, x_time)  # (B, T, d_model)
        x = x.transpose(1, 2)  # (B, d_model, T)
        x = self.pool(x).squeeze(-1)  # (B, d_model)
        return self.classifier(x)

def print_random_examples(model, dataset, device, n=3):
    model.eval()
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    for i, idx in enumerate(indices):
        X, y = dataset[idx]
        X = X.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(X)
            pred = out.argmax(dim=1).item()
        pred_label = INV_LABEL_MAP[pred]
        true_label = INV_LABEL_MAP[y.item()]
        print(f"Random Example {i+1}: Predicted={pred_label}, True={true_label}")

def main(batch_size=16, epochs=10, lr=1e-3, val_split=0.2):
    # Settings
    train_dirs = ["data/IRMAS-TrainingData", "data/solo instruments/processed"]
    device = select_device()
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size} | Epochs: {epochs} | Learning rate: {lr}")

    ensure_model_dir()
    # Precompute features if not already done
    if not (os.path.exists("model/irmas_train_features.pt") and os.path.exists("model/irmas_train_labels.pt")):
        precompute_features(train_dirs, LABEL_MAP, "irmas_train")

    # Load precomputed dataset
    features = torch.load("model/irmas_train_features.pt")
    labels = torch.load("model/irmas_train_labels.pt")
    print(f"Loaded {features.shape[0]} total samples")

    # Split into train/val
    num_samples = features.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split = int(num_samples * (1 - val_split))
    train_idx, val_idx = indices[:split], indices[split:]
    train_features, train_labels = features[train_idx], labels[train_idx]
    val_features, val_labels = features[val_idx], labels[val_idx]

    # Create datasets and dataloaders
    train_dataset = IRMASPrecomputedDataset(train_features, train_labels)
    val_dataset = IRMASPrecomputedDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Use the improved model
    # model = ImprovedCNN(len(LABEL_MAP)).to(device)
    model = SimpleCNN(len(LABEL_MAP)).to(device)
    # model = HybridTransformerInstrumentClassifier(len(LABEL_MAP)).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    print("\nStarting training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss = train(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Validate
        val_acc = evaluate(model, val_loader, device)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "model/cnn_gen.pt")
            print(f"✅ New best model saved! Accuracy: {val_acc:.4f}")
        
        # Print some random examples
        if epoch % 3 == 0:
            print_random_examples(model, val_dataset, device)
    
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Final evaluation
    print("\nFinal evaluation on validation set:")
    final_acc = evaluate(model, val_loader, device)
    print(f"Final accuracy: {final_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()
    
    main(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, val_split=args.val_split) 