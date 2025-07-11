from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import torchaudio
import librosa
import tempfile
import os
from autoencoder import Autoencoder
from cnn_gen import SimpleCNN, LABEL_MAP, INV_LABEL_MAP, select_device
from fastapi.responses import FileResponse, JSONResponse
import glob

# --- Config ---
SR = 22050
N_MELS = 80
LATENT_DIM = 512
CHUNK_DURATION = 3.0
AUTOENCODER_CKPT = 'checkpoints/autoencoder_best.pth'

app = FastAPI()

# Allow CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder = Autoencoder(latent_dim=LATENT_DIM, n_mels=N_MELS).to(device)
autoencoder.load_state_dict(torch.load(AUTOENCODER_CKPT, map_location=device))
autoencoder.eval()

# Load instrument classification model
instrument_device = select_device()
instrument_model = SimpleCNN(len(LABEL_MAP)).to(instrument_device)
instrument_model.load_state_dict(torch.load("cnn_gen.pt", map_location=instrument_device))
instrument_model.eval()

# Instrument names mapping
INSTRUMENT_NAMES = {
    "cel": "🎻 Cello",
    "cla": "🎵 Clarinet", 
    "flu": "🎶 Flute",
    "gac": "🎸 Acoustic Guitar",
    "gel": "🎸 Electric Guitar",
    "org": "🎹 Organ",
    "pia": "🎹 Piano",
    "sax": "🎷 Saxophone",
    "tru": "🎺 Trumpet",
    "vio": "🎻 Violin",
    "voi": "🎤 Voice"
}

def chunk_to_logmel(y, sr=SR, n_mels=N_MELS, chunk_duration=CHUNK_DURATION):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=256, win_length=1024, fmin=0, fmax=8000, power=1.0
    )
    mel_log = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    mel_log = np.clip(mel_log, a_min=-11.5129, a_max=2.0)
    mel_log = mel_log[np.newaxis, :, :]
    return mel_log

def pad_or_crop_mel(mel, target_shape=(1, 80, 129)):
    c, h, w = mel.shape
    tc, th, tw = target_shape
    if w < tw:
        pad_width = tw - w
        mel = np.pad(mel, ((0,0), (0,0), (0, pad_width)), mode='constant')
    elif w > tw:
        mel = mel[:, :, :tw]
    if h < th:
        pad_height = th - h
        mel = np.pad(mel, ((0,0), (0, pad_height), (0,0)), mode='constant')
    elif h > th:
        mel = mel[:, :th, :]
    return mel

class StreamingClassifier:
    def __init__(self, model, device, window_duration=3.0, sr=16000):
        self.model = model
        self.device = device
        self.window_duration = window_duration
        self.sr = sr
        self.samples_per_window = int(sr * window_duration)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_mels=128, n_fft=2048, hop_length=512
        )
        
    def preprocess_audio_segment(self, audio_segment):
        if len(audio_segment) < self.samples_per_window:
            audio_segment = np.pad(audio_segment, (0, self.samples_per_window - len(audio_segment)))
        elif len(audio_segment) > self.samples_per_window:
            audio_segment = audio_segment[:self.samples_per_window]
        
        waveform = torch.tensor(audio_segment, dtype=torch.float32).unsqueeze(0)
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log2(mel_spec + 1e-8)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
        return mel_spec.unsqueeze(0)
    
    def predict_instrument(self, audio_segment):
        try:
            mel_spec = self.preprocess_audio_segment(audio_segment)
            mel_spec = mel_spec.to(self.device)
            
            with torch.no_grad():
                logits = self.model(mel_spec)
                probabilities = torch.softmax(logits, dim=1)
                pred_idx = logits.argmax(dim=1).item()
                confidence = probabilities.max(dim=1)[0].item()
                pred_label = INV_LABEL_MAP[pred_idx]
                all_probs = probabilities.cpu().numpy()[0]
                
            return pred_label, confidence, all_probs
        except Exception as e:
            return "Error", 0.0, np.zeros(len(LABEL_MAP))

@app.post("/api/encode")
async def encode_audio(file: UploadFile = File(...)):
    # Save uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Load audio
    y, sr = librosa.load(tmp_path, sr=SR, mono=True)
    os.remove(tmp_path)

    # Take the first CHUNK_DURATION seconds (or pad if too short)
    n_samples = int(CHUNK_DURATION * SR)
    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)), mode='constant')
    else:
        y = y[:n_samples]

    # Convert to log-mel
    mel = chunk_to_logmel(y, sr=SR, n_mels=N_MELS, chunk_duration=CHUNK_DURATION)
    mel = pad_or_crop_mel(mel, target_shape=(1, N_MELS, 129))
    mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 1, 80, 129)

    # Encode
    with torch.no_grad():
        z, skips = autoencoder.encoder(mel_tensor)
        z_np = z.cpu().numpy().flatten()  # (latent_dim,)

    return {"latent": z_np.tolist()}

@app.post("/api/classify")
async def classify_instruments(
    file: UploadFile = File(...),
    window_duration: float = Form(3.0)
):
    """Classify instruments in audio file with time-window analysis"""
    
    # Save uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name
    
    try:
        # Load audio
        audio_data, sr = librosa.load(temp_path, sr=16000, mono=True)
        os.remove(temp_path)
        
        # Initialize classifier
        classifier = StreamingClassifier(instrument_model, instrument_device, window_duration)
        
        # Analyze in windows
        total_duration = len(audio_data) / classifier.sr
        num_windows = int(total_duration / window_duration)
        results = []
        
        for i in range(num_windows):
            start_sample = i * classifier.samples_per_window
            end_sample = start_sample + classifier.samples_per_window
            
            if end_sample > len(audio_data):
                break
            
            window = audio_data[start_sample:end_sample]
            pred_label, confidence, all_probs = classifier.predict_instrument(window)
            
            start_time = i * window_duration
            end_time = (i + 1) * window_duration
            
            result = {
                "window": i + 1,
                "start_time": start_time,
                "end_time": end_time,
                "instrument": pred_label,
                "instrument_name": INSTRUMENT_NAMES.get(pred_label, pred_label.upper()),
                "confidence": confidence,
                "probabilities": all_probs.tolist()
            }
            
            results.append(result)
        
        # Calculate summary statistics
        instruments = [r["instrument"] for r in results]
        most_common = max(set(instruments), key=instruments.count) if instruments else "unknown"
        avg_confidence = np.mean([r["confidence"] for r in results]) if results else 0.0
        
        summary = {
            "most_common_instrument": most_common,
            "most_common_instrument_name": INSTRUMENT_NAMES.get(most_common, most_common.upper()),
            "average_confidence": avg_confidence,
            "total_windows": len(results),
            "total_duration": total_duration
        }
        
        return {
            "summary": summary,
            "results": results
        }
        
    except Exception as e:
        if "temp_path" in locals():
            os.remove(temp_path)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/remix")
async def remix_audio(
    file1: UploadFile = File(...),  # User's uploaded hiphop audio
    parent_track_filename: str = Form(...),  # e.g., 'blues.00034.wav'
    remix_mode: str = Form("interleave"),
    crossfade_duration: float = Form(0.1),
    alpha: float = Form(1.0),
    transition_chunks: int = Form(3)
):
    import tempfile
    import os
    from remix_mixtape import main as remix_main

    # Map frontend remix_mode to backend
    backend_mode = remix_mode
    if remix_mode == "blend":
        backend_mode = "weighted_blend"
    # 'interleave' and 'average' are passed as is

    # Save user's uploaded file to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp1:
        tmp1.write(await file1.read())
        user_wav_path = tmp1.name

    # Path to the original genre .wav file
    genre_wav_path = os.path.join("Data/genres_original", parent_track_filename)
    if not os.path.exists(genre_wav_path):
        os.remove(user_wav_path)
        return JSONResponse(status_code=404, content={"error": f"Track not found: {parent_track_filename}"})

    # Output file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp2:
        output_wav_path = tmp2.name

    try:
        # Always pass genre_wav_path as latents1 (dataset), user_wav_path as latents2 (user upload)
        remix_main(
            genre_wav_path,  # dataset track (latents1)
            user_wav_path,   # user-uploaded track (latents2)
            output_wav=output_wav_path,
            remix_mode=backend_mode,
            crossfade_duration=crossfade_duration,
            alpha=alpha
        )
    except Exception as e:
        os.remove(user_wav_path)
        os.remove(output_wav_path)
        return JSONResponse(status_code=500, content={"error": str(e)})

    os.remove(user_wav_path)
    return FileResponse(output_wav_path, media_type="audio/wav", filename="mixtape.wav")

# To run: uvicorn audio_backend:app --reload