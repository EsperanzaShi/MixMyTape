import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from dataset import SpectrogramDataset
from augment import aggressive_augment_spectrogram
from model import GenreCNN
import os
from tqdm import tqdm
import wandb
import numpy as np
import sklearn.metrics
import random
import librosa

# Config
DATA_DIR = 'Data/data_preprocessed/chunks_3s/train'
VAL_DIR = 'Data/data_preprocessed/chunks_3s/val'
GENRE_IDX_PATH = 'Data/data_preprocessed/genre_to_idx.json'
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# wandb init
wandb.init(
    project="genre-classification-3s-chunks",  # or your desired project name
    entity="week5"  # this should match your team/org name on wandb
)

# Load genre mapping
with open(GENRE_IDX_PATH) as f:
    genre_to_idx = json.load(f)
    idx_to_genre = {v: k for k, v in genre_to_idx.items()}

# Datasets and loaders
train_dataset = SpectrogramDataset(DATA_DIR, genre_to_idx, augment=aggressive_augment_spectrogram)
val_dataset = SpectrogramDataset(VAL_DIR, genre_to_idx)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model, loss, optimizer
model = GenreCNN(num_classes=len(genre_to_idx)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Per-class accuracy function
def per_class_accuracy(val_labels, val_preds, idx_to_genre):
    val_labels = np.array(val_labels)
    val_preds = np.array(val_preds)
    class_acc = {}
    for idx, genre in idx_to_genre.items():
        mask = val_labels == idx
        if np.sum(mask) > 0:
            class_acc[genre] = np.mean(val_preds[mask] == idx)
        else:
            class_acc[genre] = float('nan')
    return class_acc

# Training loop with tqdm

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in tqdm(loader, desc='Training', leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    acc = correct / total if total > 0 else 0
    return running_loss / len(loader), acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for xb, yb in tqdm(loader, desc='Evaluating', leave=False):
        xb, yb = xb.to(device), yb.to(device)
        with torch.no_grad():
            out = model(xb)
            loss = criterion(out, yb)
            running_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())
    acc = correct / total if total > 0 else 0
    return running_loss / len(loader), acc, torch.cat(all_preds), torch.cat(all_labels)

for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, DEVICE)
    # Per-class accuracy
    class_acc = per_class_accuracy(val_labels.cpu().numpy(), val_preds.cpu().numpy(), idx_to_genre)
    # Learning rate
    current_lr = optimizer.param_groups[0]['lr']
    # Log to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "learning_rate": current_lr,
        **{f"val_acc_{genre}": acc for genre, acc in class_acc.items()}
    })
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    n_vis = min(5, len(val_preds))
    if n_vis > 0:
        print("Sample predictions (predicted | true):")
        for i in range(n_vis):
            pred_idx = val_preds[i].item()
            true_idx = val_labels[i].item()
            print(f"  {idx_to_genre[pred_idx]} | {idx_to_genre[true_idx]}")
    else:
        print("No validation samples to visualize.")
    # Step the scheduler
    scheduler.step(val_loss)

# After all epochs: log confusion matrix and audio samples
val_preds_np = val_preds.cpu().numpy()
val_labels_np = val_labels.cpu().numpy()
wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
    preds=val_preds_np, y_true=val_labels_np, class_names=[idx_to_genre[i] for i in range(len(idx_to_genre))]
)})

# Audio logging for 3 random samples from validation set
sample_indices = random.sample(range(len(val_dataset)), 3)
for i in sample_indices:
    npy_path = val_dataset.filepaths[i]
    genre = os.path.basename(npy_path).split('.')[0]
    base = '.'.join(os.path.basename(npy_path).split('_chunk')[0].split('.'))
    wav_path = f'Data/genres_original/{genre}/{base}.wav'
    if os.path.exists(wav_path):
        y, sr = librosa.load(wav_path, sr=None)
        pred_idx = val_preds_np[i]
        true_idx = val_labels_np[i]
        wandb.log({f"audio_sample_{i}": wandb.Audio(y, sample_rate=sr, caption=f"Pred: {idx_to_genre[pred_idx]}, True: {idx_to_genre[true_idx]}")})

# Save pretrained weights
os.makedirs('checkpoints', exist_ok=True)
torch.save(model.state_dict(), 'checkpoints/genre_cnn_pretrained_3schunks.pth')
wandb.save('checkpoints/genre_cnn_pretrained_3schunks.pth')
print("Pretrained weights saved to checkpoints/genre_cnn_pretrained_3schunks.pth")
