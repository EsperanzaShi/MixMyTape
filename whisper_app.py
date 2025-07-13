import torch
import whisper
from pynput import keyboard
import pyautogui
import sounddevice as sd
import numpy as np
import webbrowser
import time
import re

def execute_command(transcription):
    text = transcription.lower().strip()

    # Pattern 1: open browser and search google for X
    match_search = re.search(
        r"(?:search google(?: for)?|google for|search for)\s+(.+)", text, re.IGNORECASE
    )
    if match_search:
        query = match_search.group(1).strip()
        url = "https://www.google.com"
        webbrowser.open(url)
        time.sleep(1.5)
        pyautogui.typewrite(query)
        time.sleep(0.3)
        pyautogui.press('enter')
        return

    # Pattern 2: open chatgpt and ask it Y
    match_chatgpt = re.search(
        r"ask\s+chat\s*gpt(?:\s*to)?\s+(.+)|ask\s+chatgpt(?:\s*to)?\s+(.+)", text, re.IGNORECASE
    )
    if match_chatgpt:
        prompt = match_chatgpt.group(1).strip()
        webbrowser.open("https://chatgpt.com/?temporary-chat=true")
        time.sleep(2)
        pyautogui.typewrite(prompt)
        time.sleep(0.5)
        pyautogui.press('enter')
        return
    
    # Pattern 3: open browser to demo the app
    match_open = re.search(
        r"(?:open\s+(?:the\s+)?(?:app|demo))", text, re.IGNORECASE
    )
    if match_open:
        url = "http://localhost:3000"
        webbrowser.open(url)
        time.sleep(1.5)
        return

    print("Sorry, command not recognised.")


model = whisper.load_model("tiny")
checkpoint = torch.load("whisper_tiny_finetuned.pt", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

sample_rate = 16000
audio_data = []
recording = False
stream = None

def callback(indata, frames, time, status):
    if recording:
        audio_data.append(indata.copy())

def on_press(key):
    global recording, stream, audio_data
    try:
        if key == keyboard.Key.ctrl and not recording:
            print("Recording... (release CTRL to stop)")
            recording = True
            audio_data = []
            stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', callback=callback)
            stream.start()
    except Exception as e:
        print("Error:", e)

def on_release(key):
    global recording, stream, audio_data
    if key == keyboard.Key.ctrl and recording:
        print("Stopped recording.")
        recording = False
        stream.stop()
        stream.close()
        audio = np.concatenate(audio_data, axis=0).flatten()
        print("Transcribing...")
        result = model.transcribe(audio, fp16=False, language="en")
        print("Transcription:", result["text"])
        execute_command(result["text"])
        return False  # End listener after one recording, or remove to keep running

while True:
    print("Hold CTRL to record...")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
    print("\n")