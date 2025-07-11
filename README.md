# AI-Assisted Music Remix Web App

An interactive web application for AI-assisted music remixing using deep learning, UMAP visualization, and a modern React UI. Users can upload hiphop tracks, explore similar audio chunks in 3D space, and generate mixtapes using various remix modes.

## Features

- **3D Audio Visualization**: Interactive UMAP plot showing audio chunks from GTZAN dataset
- **Genre Filtering**: Filter samples by blues, classical, and jazz genres
- **Top-N Similarity**: Find the most similar audio chunks to uploaded tracks
- **Audio Preview**: Listen to dataset chunks and generated remixes
- **Multiple Remix Modes**: interleave, blend, and average modes
- **Real-time Processing**: FastAPI backend for audio encoding and remixing

## Prerequisites

- Python 3.13+
- Node.js 18+ and npm
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

## Required Files and Data

### Essential Data Files
```
audioML/
├── latents_blues_classical_jazz_meta.json    # UMAP embeddings + metadata (264KB) 
├── latents_blues_classical_jazz.npy          # Latent vectors (5.9MB)
├── checkpoints/
│   └── autoencoder_best.pth                  # Trained autoencoder model
├── hifi-gan/
│   └── UNIVERSAL_V1/
│       ├── g_02500000                        # HiFi-GAN generator copied from their github
│       └── config.json                       # HiFi-GAN config copied from their github
└── public/
    └── data/
        └── umap_embeddings.json              # created by umap_embed_and_merge.py
```

### Required Directories
```
audioML/
└── Data/                                      # Original full-length tracks
    └── genre_original
        ├── blues/
        ├── classical/
        └── jazz/

└── my-mixtape-ui
    └── public
        └── chunked_wavs/                      # 3-second audio chunks for preview
            ├── blues/
            ├── classical/
            └── jazz/


## Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd audioML
```

### 2. Install Python Dependencies
```bash
uv sync
```

### 3. Install Frontend Dependencies
```bash
cd my-mixtape-ui
npm install
```

## Running the Application

### 1. Start the Backend (FastAPI) (Split terminal)
```bash
# From the root directory
cd audioML
uvicorn audio_backend:app --reload --host 0.0.0.0 --port 8000
```

The backend provides:
- `/api/encode` - Encode uploaded audio to latent vectors
- `/api/remix` - Generate remixes using selected mode

### 2. Start the Frontend (React)
```bash
# From the my-mixtape-ui directory
cd my-mixtape-ui
npm start
```

The React app will open at `http://localhost:3000`

## Usage

### 1. Upload Hiphop Track
- Click "Choose File" in the Track section
- Select a hiphop audio file (.wav, .mp3, etc.)
- The track will be encoded and appear as a white anchor point in the 3D plot

### 2. Explore Similar Samples
- Use the genre checkboxes to filter samples
- Adjust the "Top-N recommend" slider to see more/less similar points
- Red points indicate the most similar chunks to your uploaded track
- Click on red points to preview the audio (plays at 30% volume)

### 3. Generate Remix
- Select a sample by clicking on it
- Choose a remix mode:
  - **interleave**: Alternates between your track and dataset track
  - **blend**: Weighted blend of both tracks
  - **average**: Equal mix of both tracks
- Click the mode button to generate the mixtape
- Play or download the generated remix

## File Structure

```
audioML/
├── audio_backend.py              # FastAPI backend server
├── remix_mixtape.py             # Core remixing logic
├── autoencoder.py               # Autoencoder model definition
├── audio_utils.py               # Audio processing utilities
├── extract_latents.py           # Extract latent vectors from audio
├── umap_embed_and_merge.py      # UMAP dimensionality reduction
├── export_chunked_wavs.py       # Export 3s audio chunks
├── train_autoencoder.py         # Train the autoencoder model
├── my-mixtape-ui/               # React frontend
│   ├── src/
│   │   ├── App.js               # Main React component
│   │   └── App.css              # Styling
│   └── public/
│       └── data/                # Frontend data files
├── checkpoints/                 # Model checkpoints
├── hifi-gan/                    # HiFi-GAN vocoder
├── chunked_wavs/                # 3s audio chunks
├── playlist/                    # Original tracks
└── pyproject.toml              # Python dependencies
```

## Data Processing Pipeline

1. **Audio Preprocessing**: GTZAN dataset chunks → 3s segments → Mel spectrograms
2. **Model Training**: Autoencoder trained on Mel spectrograms
3. **Latent Extraction**: Extract latent vectors for blues, classical, jazz chunks
4. **UMAP Embedding**: Reduce latents to 3D for visualization
5. **Frontend Integration**: Load embeddings + metadata for interactive UI

## Troubleshooting

### Missing Data Files
If you get errors about missing files:
- Ensure `latents_blues_classical_jazz_meta.json` exists in the root directory
- Copy it to `my-mixtape-ui/public/data/umap_embeddings.json`
- Verify `chunked_wavs/` directory contains audio chunks

### Backend Connection Issues
- Check that the backend is running on port 8000
- Ensure CORS is properly configured in `audio_backend.py`
- Verify all model checkpoints are present

### Audio Playback Issues
- Dataset chunks play at 30% volume to match remix loudness
- Ensure audio files are in supported formats (.wav, .mp3)

## Development

### Adding New Genres
1. Add audio files to `playlist/[genre]/`
2. Run `export_chunked_wavs.py` to create chunks
3. Run `extract_latents.py` to extract latents
4. Update UMAP embeddings with `umap_embed_and_merge.py`

### Modifying Remix Modes
Edit `remix_mixtape.py` to add new remix algorithms or modify existing ones.

## License

[Add your license information here]
