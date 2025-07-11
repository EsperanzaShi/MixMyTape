import os
import numpy as np
import matplotlib.pyplot as plt
import json
import random

# Paths
base_dir = 'Data/data_preprocessed/chunks_3s'
splits = ['train', 'val', 'test']
with open('Data/data_preprocessed/genre_to_idx.json') as f:
    genre_to_idx = json.load(f)

for split in splits:
    print(f"\n=== {split.upper()} SPLIT ===")
    split_dir = os.path.join(base_dir, split)
    # Find all files for each genre
    genre_files = {genre: [] for genre in genre_to_idx}
    for fname in os.listdir(split_dir):
        if fname.endswith('.npy'):
            genre = fname.split('.')[0]
            if genre in genre_files:
                genre_files[genre].append(os.path.join(split_dir, fname))
    # Visualize one random sample per genre
    for genre, files in genre_files.items():
        if files:
            file_path = random.choice(files)
            arr = np.load(file_path)
            plt.figure(figsize=(6, 3))
            plt.imshow(arr[0], aspect='auto', origin='lower')
            plt.colorbar()
            plt.title(f"{split} - {genre}\n{os.path.basename(file_path)}")
            plt.xlabel('Time')
            plt.ylabel('Mel bands')
            plt.tight_layout()
            plt.show()
        else:
            print(f"No files found for genre {genre} in {split} split.")
