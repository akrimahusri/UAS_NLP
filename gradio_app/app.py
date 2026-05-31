import os
import tempfile
import sys
from pathlib import Path

import gradio as gr
import scipy.io.wavfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline import process_audio_bytes

def _load_audio_bytes(audio):
    if audio is None:
        return None, None

    if isinstance(audio, str):
        audio_path = audio
        with open(audio_path, "rb") as f:
            return f.read(), Path(audio_path).name

    if isinstance(audio, tuple) and len(audio) == 2:
        sr, audio_data = audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            scipy.io.wavfile.write(tmpfile.name, sr, audio_data)
            audio_path = tmpfile.name

        with open(audio_path, "rb") as f:
            return f.read(), Path(audio_path).name

    return None, None


def voice_chat(mic_audio, uploaded_audio, response_mode):
    audio = uploaded_audio or mic_audio
    file_bytes, file_name = _load_audio_bytes(audio)
    if file_bytes is None:
        return None, "", "", ""

    result = process_audio_bytes(file_bytes, file_name or "voice.wav", response_mode=response_mode)
    return (
        result.audio_output if result.audio_output else None,
        result.transcript,
        result.normalized_text or "",
        result.response_text,
    )

# UI Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ Voice Chatbot")
    gr.Markdown("Kirim audio dari mikrofon atau upload file, lalu dapatkan balasan suara dari asisten AI.")

    with gr.Row():
        with gr.Column():
            mic_input = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Rekam dari mikrofon")
            upload_input = gr.Audio(sources=["upload"], type="filepath", label="📁 Upload file audio")
            mode_input = gr.Radio(
                choices=["preserve_cs", "normalized"],
                value="preserve_cs",
                label="Mode respons",
            )
            submit_btn = gr.Button("🔁 Submit")
        with gr.Column():
            audio_output = gr.Audio(type="filepath", label="🔊 Balasan dari Asisten")

    transcript_output = gr.Textbox(label="Hasil STT", lines=3)
    normalized_output = gr.Textbox(label="Normalisasi / Tagging", lines=3)
    response_output = gr.Textbox(label="Jawaban LLM", lines=4)

    submit_btn.click(
        fn=voice_chat,
        inputs=[mic_input, upload_input, mode_input],
        outputs=[audio_output, transcript_output, normalized_output, response_output],
    )


if __name__ == "__main__":
    demo.launch()
