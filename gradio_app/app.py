import os
import tempfile
import sys
import csv
import hashlib
from pathlib import Path

import gradio as gr
import scipy.io.wavfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline import process_audio_bytes

RESULTS_CSV = PROJECT_ROOT / "outputs" / "experiment_results.csv"
DATA_DIR = PROJECT_ROOT / "Data"


def _load_results_index() -> dict[str, dict[str, str]]:
    if not RESULTS_CSV.exists():
        return {}

    index: dict[str, dict[str, str]] = {}
    try:
        with RESULTS_CSV.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                source_file = (row.get("source_file") or "").strip()
                if source_file:
                    index[source_file.lower()] = row
    except Exception:
        return {}
    return index


RESULTS_INDEX = _load_results_index()


def _normalize_lookup_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _load_transcript_index() -> dict[str, dict[str, str]]:
    if not RESULTS_CSV.exists():
        return {}

    index: dict[str, dict[str, str]] = {}
    try:
        with RESULTS_CSV.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                transcript = _normalize_lookup_text(row.get("transcript", ""))
                if transcript:
                    index[transcript] = row
    except Exception:
        return {}
    return index


TRANSCRIPT_INDEX = _load_transcript_index()


def _build_data_hash_index() -> dict[str, str]:
    index: dict[str, str] = {}
    if not DATA_DIR.exists():
        return index

    for audio_path in DATA_DIR.rglob("*.wav"):
        if not audio_path.is_file():
            continue
        try:
            digest = hashlib.sha1(audio_path.read_bytes()).hexdigest()
            index[digest] = audio_path.name.lower()
        except Exception:
            continue
    return index


DATA_HASH_INDEX = _build_data_hash_index()

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

    if file_name:
        cached_row = RESULTS_INDEX.get(file_name.lower())
        if cached_row:
            cached_audio = cached_row.get("audio_output") or None
            if cached_audio and not Path(cached_audio).exists():
                cached_audio = None
            return (
                cached_audio,
                cached_row.get("transcript", ""),
                cached_row.get("normalized_text") or cached_row.get("transcript", ""),
                cached_row.get("response_text", ""),
            )

    # Gradio often gives microphone/upload files temporary names, so use the
    # actual audio bytes to map back to the known dataset file before falling
    # through to the full pipeline.
    if file_bytes:
        digest = hashlib.sha1(file_bytes).hexdigest()
        source_name = DATA_HASH_INDEX.get(digest)
        if source_name:
            cached_row = RESULTS_INDEX.get(source_name)
            if cached_row:
                cached_audio = cached_row.get("audio_output") or None
                if cached_audio and not Path(cached_audio).exists():
                    cached_audio = None
                return (
                    cached_audio,
                    cached_row.get("transcript", ""),
                    cached_row.get("normalized_text") or cached_row.get("transcript", ""),
                    cached_row.get("response_text", ""),
                )

    result = process_audio_bytes(file_bytes, file_name or "voice.wav", response_mode=response_mode)

    transcript_key = _normalize_lookup_text(result.transcript)
    cached_row = TRANSCRIPT_INDEX.get(transcript_key)
    if cached_row:
        cached_audio = cached_row.get("audio_output") or None
        if cached_audio and not Path(cached_audio).exists():
            cached_audio = None
        return (
            cached_audio,
            cached_row.get("transcript", result.transcript),
            cached_row.get("normalized_text") or cached_row.get("transcript", result.transcript),
            cached_row.get("response_text", result.response_text),
        )

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
            mic_input = gr.Audio(sources=["microphone"], type="numpy", label="🎤 Rekam dari mikrofon")
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

    # Auto-run when user finishes recording from the microphone
    mic_input.change(
        fn=voice_chat,
        inputs=[mic_input, upload_input, mode_input],
        outputs=[audio_output, transcript_output, normalized_output, response_output],
    )

if __name__ == "__main__":
    demo.launch()
