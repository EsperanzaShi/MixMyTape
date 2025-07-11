import numpy as np
import json
import umap
from umap import UMAP
import os

# --- Config ---
LATENTS_PATH = 'latents_blues_classical_jazz.npy'
META_PATH = 'latents_blues_classical_jazz_meta.json'
OUTPUT_JSON = 'public/data/umap_embeddings.json'  # Adjust as needed for your UI
N_COMPONENTS = 3

# --- Load data ---
latents = np.load(LATENTS_PATH)
with open(META_PATH) as f:
    meta = json.load(f)

# --- Run UMAP ---
umap_model = UMAP(n_components=N_COMPONENTS, n_neighbors=5, min_dist=0.5, metric='cosine', random_state=42)
embedding = umap_model.fit_transform(latents)

# --- Merge and save ---
points = []
for i, m in enumerate(meta):
    point = {
        'x': float(embedding[i, 0]),
        'y': float(embedding[i, 1]),
        'z': float(embedding[i, 2]),
        'genre': m['genre'],
        'path': m['path'],  # This should be a relative path to the .wav file for the UI
        'latent': latents[i].tolist(),
        'id': i
    }
    points.append(point)

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, 'w') as f:
    json.dump(points, f, indent=2)
print(f"Saved {len(points)} points to {OUTPUT_JSON}") 