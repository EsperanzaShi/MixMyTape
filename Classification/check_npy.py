import os
import numpy as np

def check_npy_files(directory, num_files=5):
    checked = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                path = os.path.join(root, file)
                arr = np.load(path)
                print(f"{file}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, finite={np.isfinite(arr).all()}")
                if not np.isfinite(arr).all():
                    print(f"  [Warning] {file} contains NaN or Inf values!")
                checked += 1
                if checked >= num_files:
                    return

# Example usage:
print("Checking full_30s/train:")
check_npy_files('Data/data_preprocessed/full_30s/train')

print("\nChecking chunks_3s/train:")
check_npy_files('Data/data_preprocessed/chunks_3s/train')
