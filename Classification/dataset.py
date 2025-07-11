import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    def __init__(self, data_dir, genre_to_idx, augment=None):
        self.filepaths = []
        self.labels = []
        self.augment = augment
        for fname in os.listdir(data_dir):
            if fname.endswith('.npy'):
                self.filepaths.append(os.path.join(data_dir, fname))
                genre = fname.split('.')[0]
                self.labels.append(genre_to_idx[genre])
    def __len__(self):
        return len(self.filepaths)
    def __getitem__(self, idx):
        x = np.load(self.filepaths[idx]).astype(np.float32)
        if self.augment:
            x = self.augment(x)
        x = torch.from_numpy(x).float()
        y = self.labels[idx]
        return x, y

if __name__ == '__main__':
    import json
    from collections import Counter
    try:
        import matplotlib.pyplot as plt
        has_plt = True
    except ImportError:
        has_plt = False

    GENRE_IDX_PATH = 'Data/data_preprocessed/genre_to_idx.json'
    DATA_DIR = 'Data/data_preprocessed/chunks_3s/train'

    with open(GENRE_IDX_PATH) as f:
        genre_to_idx = json.load(f)
    print('genre_to_idx mapping:')
    print(genre_to_idx)

    dataset = SpectrogramDataset(DATA_DIR, genre_to_idx)
    print('\nSample file-label-genre pairs:')
    for i in range(5):
        _, label = dataset[i]
        genre = list(genre_to_idx.keys())[list(genre_to_idx.values()).index(label)]
        print(f"File: {dataset.filepaths[i]}, Label index: {label}, Genre: {genre}")

    labels = [label for _, label in dataset]
    label_counts = Counter(labels)
    print('\nClass distribution:')
    for idx, count in label_counts.items():
        genre = list(genre_to_idx.keys())[list(genre_to_idx.values()).index(idx)]
        print(f"{genre}: {count} samples")

    if has_plt:
        plt.bar([list(genre_to_idx.keys())[i] for i in label_counts.keys()], label_counts.values())
        plt.xticks(rotation=45)
        plt.ylabel('Number of samples')
        plt.title('Class distribution')
        plt.tight_layout()
        plt.show()
    else:
        print("matplotlib not installed, skipping plot.")
