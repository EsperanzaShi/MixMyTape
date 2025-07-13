import streamlit as st
import torch
import torchaudio
import numpy as np
import time
import tempfile
import os
import plotly.graph_objects as go
import plotly.express as px
import soundfile as sf
from cnn_gen import SimpleCNN, LABEL_MAP, INV_LABEL_MAP, select_device

# Page configuration
st.set_page_config(page_title="🎵 Instrument Classifier", page_icon="🎵", layout="wide")

# Proper instrument names mapping
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

@st.cache_resource
def load_model():
    device = select_device()
    model = SimpleCNN(len(LABEL_MAP)).to(device)
    model.load_state_dict(torch.load("cnn_gen.pt", map_location=device))
    model.eval()
    return model, device

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
            st.error(f"Error in prediction: {e}")
            return "Error", 0.0, np.zeros(len(LABEL_MAP))

def main():
    st.title("🎵 Real-Time Instrument Classifier")
    
    # Load model
    with st.spinner("Loading model..."):
        model, device = load_model()
        st.success(f"✅ Model loaded on {device}")
    
    # Settings
    window_duration = st.sidebar.slider("Window Duration (seconds)", 1.0, 5.0, 3.0, 0.5)
    classifier = StreamingClassifier(model, device, window_duration)
    
    # File upload
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac", "ogg"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        try:
            audio_data, sr = sf.read(temp_path)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            if sr != classifier.sr:
                ratio = classifier.sr / sr
                new_length = int(len(audio_data) * ratio)
                audio_data = np.interp(np.linspace(0, len(audio_data), new_length), 
                                     np.arange(len(audio_data)), audio_data)
            
            # Audio info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duration", f"{len(audio_data)/classifier.sr:.1f}s")
            with col2:
                st.metric("Sample Rate", f"{classifier.sr}Hz")
            with col3:
                st.metric("Channels", "Mono")
            
            # Audio player
            st.audio(uploaded_file, format="audio/wav")
            
            # Live prediction display area - create placeholders
            st.markdown("---")
            st.subheader("🎼 Live Prediction")
            
            # Create placeholders for live updates
            live_prediction_placeholder = st.empty()
            live_confidence_placeholder = st.empty()
            live_progress_placeholder = st.empty()
            live_time_placeholder = st.empty()
            
            if st.button("🚀 Start Analysis", type="primary"):
                if "analysis_results" not in st.session_state:
                    st.session_state.analysis_results = []
                
                total_duration = len(audio_data) / classifier.sr
                num_windows = int(total_duration / window_duration)
                
                st.subheader("🎼 Analysis Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(num_windows):
                    start_sample = i * classifier.samples_per_window
                    end_sample = start_sample + classifier.samples_per_window
                    
                    if end_sample > len(audio_data):
                        break
                    
                    window = audio_data[start_sample:end_sample]
                    pred_label, confidence, all_probs = classifier.predict_instrument(window)
                    
                    progress = (i + 1) / num_windows
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing window {i+1}/{num_windows}...")
                    
                    start_time = i * window_duration
                    end_time = (i + 1) * window_duration
                    
                    result = {
                        "window": i + 1,
                        "start_time": start_time,
                        "end_time": end_time,
                        "instrument": pred_label,
                        "confidence": confidence,
                        "probabilities": all_probs
                    }
                    
                    st.session_state.analysis_results.append(result)
                    
                    # Update live prediction display
                    instrument_name = INSTRUMENT_NAMES.get(pred_label, pred_label.upper())
                    
                    # Update the placeholders with new values
                    live_prediction_placeholder.markdown(f"### {instrument_name}")
                    live_confidence_placeholder.markdown(f"**Confidence:** {confidence:.1%}")
                    live_progress_placeholder.progress(confidence)
                    live_time_placeholder.markdown(f"*Time: {start_time:.1f}s - {end_time:.1f}s*")
                    
                    # Show detailed results in columns
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        st.metric("Window", f"{i+1}/{num_windows}")
                    with col2:
                        st.markdown(f"**{instrument_name}** ({start_time:.1f}s-{end_time:.1f}s)")
                        st.progress(confidence)
                    with col3:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    time.sleep(0.1)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Analysis complete!")
                st.success(f"Successfully analyzed {num_windows} windows!")
                
                os.unlink(temp_path)
        
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            if "temp_path" in locals():
                os.unlink(temp_path)
    
    # Results tab
    if "analysis_results" in st.session_state and st.session_state.analysis_results:
        st.header("📊 Results Summary")
        results = st.session_state.analysis_results
        
        instruments = [r["instrument"] for r in results]
        most_common = max(set(instruments), key=instruments.count)
        avg_confidence = np.mean([r["confidence"] for r in results])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            most_common_name = INSTRUMENT_NAMES.get(most_common, most_common.upper())
            st.metric("Most Common", most_common_name)
        with col2:
            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
        with col3:
            st.metric("Total Windows", len(results))
        
        # Results table
        import pandas as pd
        df_data = []
        for r in results:
            instrument_name = INSTRUMENT_NAMES.get(r["instrument"], r["instrument"].upper())
            df_data.append({
                "Window": r["window"],
                "Time": f"{r['start_time']:.1f}s-{r['end_time']:.1f}s",
                "Instrument": instrument_name,
                "Confidence": f"{r['confidence']:.2%}"
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
        
        if st.button("🗑️ Clear Results"):
            st.session_state.analysis_results = []
            st.rerun()

if __name__ == "__main__":
    main()
