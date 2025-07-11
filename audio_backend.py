from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import librosa
import tempfile
import os
from autoencoder import Autoencoder
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

# Load model at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder = Autoencoder(latent_dim=LATENT_DIM, n_mels=N_MELS).to(device)
autoencoder.load_state_dict(torch.load(AUTOENCODER_CKPT, map_location=device))
autoencoder.eval()

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