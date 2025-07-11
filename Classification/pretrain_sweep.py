import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import wandb
from model_sweep import GenreCNN, KERNEL_CONFIGS
from dataset import SpectrogramDataset
import os

def train_with_kernel_config(kernel_config_name, kernel_config, num_epochs=15, batch_size=32):
    """
    Train model with specific kernel configuration
    """
    # Initialize wandb with kernel config info
    run_name = f"kernel_sweep_{kernel_config_name}"
    wandb.init(
        project="audio-genre-classification",
        entity="week5",
        name=run_name,
        config={
            "kernel_config": kernel_config_name,
            "conv1_kernel": kernel_config['conv1_kernel'],
            "conv2_kernel": kernel_config['conv2_kernel'],
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": 0.001,
            "weight_decay": 1e-4
        }
    )
    
    # Load genre mapping
    with open('Data/data_preprocessed/genre_to_idx.json', 'r') as f:
        genre_to_idx = json.load(f)
    num_classes = len(genre_to_idx)
    
    # Load data
    train_dataset = SpectrogramDataset('Data/data_preprocessed/chunks_3s/train', genre_to_idx)
    val_dataset = SpectrogramDataset('Data/data_preprocessed/chunks_3s/val', genre_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model with specific kernel config
    model = GenreCNN(num_classes=num_classes, kernel_config=kernel_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        val_acc = 100. * val_correct / val_total
        scheduler.step(val_loss)
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_acc,
            'val_loss': val_loss / len(val_loader),
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        print(f'Epoch {epoch+1}/{num_epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'checkpoints/genre_cnn_{kernel_config_name}.pth')
    
    wandb.finish()
    return best_val_acc

def run_kernel_sweep():
    """
    Run training with different kernel configurations
    """
    results = {}
    
    for config_name, kernel_config in KERNEL_CONFIGS.items():
        print(f"\n{'='*50}")
        print(f"Training with kernel config: {config_name}")
        print(f"Conv1 kernel: {kernel_config['conv1_kernel']}")
        print(f"Conv2 kernel: {kernel_config['conv2_kernel']}")
        print(f"{'='*50}")
        
        try:
            best_acc = train_with_kernel_config(config_name, kernel_config, num_epochs=15)
            results[config_name] = best_acc
            print(f"Best validation accuracy for {config_name}: {best_acc:.2f}%")
        except Exception as e:
            print(f"Error training {config_name}: {e}")
            results[config_name] = 0.0
    
    # Print summary
    print(f"\n{'='*50}")
    print("KERNEL SWEEP RESULTS:")
    print(f"{'='*50}")
    for config_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{config_name}: {acc:.2f}%")
    
    # Save results
    with open('kernel_sweep_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Run the kernel sweep
    run_kernel_sweep() 