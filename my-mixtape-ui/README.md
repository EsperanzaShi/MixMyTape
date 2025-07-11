# MixMyTape UI

A React-based web interface for audio remixing and instrument classification.

## Features

### Remix Mode
- Upload hiphop tracks and find similar samples in the 3D visualization
- Generate remixes using different modes: interleave, blend, and average
- Interactive 3D plot with genre filtering and Top-N recommendations

### Instrument Classification Mode
- Upload audio files for real-time instrument classification
- Time-window analysis with configurable window duration (1-5 seconds)
- Detailed results showing instrument detection over time
- Support for 11 instrument types: Cello, Clarinet, Flute, Acoustic Guitar, Electric Guitar, Organ, Piano, Saxophone, Trumpet, Violin, and Voice

## Installation

1. Install dependencies:
```bash
npm install
```

2. Make sure the backend server is running (see main README.md for backend setup)

## Running the Application

Start the development server:
```bash
npm start
```

The app will open at `http://localhost:3000`

## Usage

### Remix Mode
1. Click the "Remix" tab
2. Upload a hiphop audio file
3. Use genre checkboxes to filter samples
4. Adjust the Top-N slider to see more/less similar points
5. Click on red points to preview audio
6. Select a sample and choose a remix mode
7. Generate and download your mixtape

### Classification Mode
1. Click the "Classify" tab
2. Upload an audio file (WAV, MP3, FLAC, etc.)
3. Adjust the window duration slider (1-5 seconds)
4. Click "Start Analysis" to begin classification
5. View results including:
   - Most common instrument detected
   - Average confidence score
   - Total number of time windows analyzed
   - Detailed time-window breakdown table

## Backend Requirements

The frontend requires the FastAPI backend to be running on `http://localhost:8000` with the following endpoints:
- `POST /api/encode` - Encode audio to latent vectors
- `POST /api/remix` - Generate remixes
- `POST /api/classify` - Classify instruments in audio

## Technical Details

- Built with React 19
- Uses Plotly.js for 3D visualization
- Responsive design with mobile support
- Real-time audio analysis with time-window processing
- Instrument classification using CNN models trained on IRMAS dataset

## File Structure

```
src/
├── App.js          # Main application component
├── App.css         # Styles for the application
└── index.js        # Application entry point
```

## Troubleshooting

- If the backend connection fails, ensure the FastAPI server is running on port 8000
- For classification issues, check that the `cnn_gen.pt` model file is present in the backend directory
- Audio file upload issues may be related to CORS settings in the backend
