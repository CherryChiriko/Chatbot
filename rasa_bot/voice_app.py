import asyncio
import os
import tempfile
import uuid
import wave
from typing import List

import requests
import streamlit as st
from faster_whisper import WhisperModel
import edge_tts
import pyaudio


# --- Config (override with env vars if needed) ---
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")

RASA_URL = os.getenv("RASA_URL", "http://localhost:5005/webhooks/rest/webhook")
RASA_SENDER = os.getenv("RASA_SENDER", "voice-ui")

FRENCH_VOICE = os.getenv("FRENCH_VOICE", "fr-FR-HenriNeural")


@st.cache_resource
def load_whisper() -> WhisperModel:
    return WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)


def record_audio(duration_seconds: int, fs: int = 16000) -> str:
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=fs,
        input=True,
        frames_per_buffer=1024,
    )
    frames: List[bytes] = []
    st.write("Enregistrement en cours... Parlez maintenant.")
    for _ in range(0, int(fs / 1024 * duration_seconds)):
        data = stream.read(1024)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

    fd, path = tempfile.mkstemp(suffix=".wav", prefix="voice_input_")
    os.close(fd)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(fs)
        wf.writeframes(b"".join(frames))
    return path


def transcribe_audio(file_path: str) -> str:
    model = load_whisper()
    segments, _ = model.transcribe(file_path, language="fr")
    return " ".join([s.text for s in segments]).strip()


def ask_rasa(text: str, sender_id: str) -> str:
    payload = {"sender": sender_id, "message": text}
    resp = requests.post(RASA_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return "Je n'ai pas de reponse pour le moment."
    # Rasa REST returns a list of messages
    parts = [m.get("text", "") for m in data if m.get("text")]
    return "\n".join(parts).strip() or "Je n'ai pas de reponse pour le moment."


async def text_to_speech_bytes(text: str) -> bytes:
    fd, path = tempfile.mkstemp(suffix=".mp3", prefix="voice_out_")
    os.close(fd)
    communicate = edge_tts.Communicate(text, FRENCH_VOICE)
    await communicate.save(path)
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    return data


st.title("Assistant Bancaire Vocal")

duration = st.slider("Duree d'enregistrement (secondes)", min_value=3, max_value=12, value=8)
sender = st.text_input("ID session (optionnel)", value=RASA_SENDER)

col1, col2 = st.columns(2)
with col1:
    start_record = st.button("Parler")
with col2:
    text_input = st.text_input("Ou tapez votre question", value="")

if start_record:
    wav_path = record_audio(duration_seconds=duration)
    try:
        user_text = transcribe_audio(wav_path)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

    st.write("Vous avez dit :", user_text if user_text else "(vide)")
    if user_text:
        response_text = ask_rasa(user_text, sender)
        st.write("Reponse :", response_text)
        audio_bytes = asyncio.run(text_to_speech_bytes(response_text))
        st.audio(audio_bytes, format="audio/mp3")

if text_input:
    response_text = ask_rasa(text_input, sender)
    st.write("Reponse :", response_text)
    audio_bytes = asyncio.run(text_to_speech_bytes(response_text))
    st.audio(audio_bytes, format="audio/mp3")
