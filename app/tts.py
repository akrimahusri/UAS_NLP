import os
import uuid
import tempfile
import subprocess
from functools import lru_cache
from pathlib import Path
import math
import wave
import struct
import shutil
import asyncio

try:
    import pyttsx3  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover - optional dependency
    pyttsx3 = None

try:
    from gtts import gTTS  # type: ignore[reportMissingImports]
except Exception:
    gTTS = None

try:
    import edge_tts  # type: ignore[reportMissingImports]
except Exception:
    edge_tts = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# path ke folder utilitas TTS
COQUI_DIR = os.path.join(BASE_DIR, "coqui_utils")

# TODO: Lengkapi jalur path ke file model TTS
# File model (misalnya checkpoint_1260000-inference.pth) harus berada di dalam folder coqui_utils/
COQUI_MODEL_PATH = os.path.join(COQUI_DIR, "checkpoint_1260000-inference.pth")

# TODO: Lengkapi jalur path ke file konfigurasi
# File config.json harus berada di dalam folder coqui_utils/
COQUI_CONFIG_PATH = os.path.join(COQUI_DIR, "config.json")

# TODO: Tentukan nama speaker yang digunakan
# Pilih nama speaker yang sesuai dengan isi file speakers.pth (misalnya: "wibowo")
COQUI_SPEAKER = "wibowo"


@lru_cache(maxsize=1)
def _load_pyttsx3_engine():
    if pyttsx3 is None:
        return None
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)
    return engine


def _tts_with_wave_fallback(text: str) -> str:
    tmp_dir = tempfile.gettempdir()
    output_path = os.path.join(tmp_dir, f"tts_{uuid.uuid4()}.wav")
    sample_rate = 22050
    base_frequency = 220.0
    amplitude = 12000
    frames: list[bytes] = []

    for index, character in enumerate(text[:240] or "silence"):
        if character.isspace():
            duration = 0.04
            tone_frequency = 0.0
        else:
            duration = 0.06
            tone_frequency = base_frequency + (ord(character) % 24) * 18.0

        total_samples = max(1, int(sample_rate * duration))
        for sample_index in range(total_samples):
            if tone_frequency == 0.0:
                sample_value = 0
            else:
                sample_value = int(
                    amplitude
                    * math.sin(2.0 * math.pi * tone_frequency * (sample_index / sample_rate))
                )
            frames.append(struct.pack("<h", sample_value))

        pause_samples = int(sample_rate * 0.015)
        frames.extend(struct.pack("<h", 0) for _ in range(pause_samples))

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(frames))

    return output_path


def _tts_with_pyttsx3(text: str, language: str = "id") -> str:
    if pyttsx3 is None:
        return _tts_with_wave_fallback(text)

    tmp_dir = tempfile.gettempdir()
    output_path = os.path.join(tmp_dir, f"tts_{uuid.uuid4()}.wav")
    engine = _load_pyttsx3_engine()
    if engine is None:
        return _tts_with_wave_fallback(text)

    # try to select an Indonesian voice if available
    try:
        voices = engine.getProperty("voices")
        chosen = None
        for v in voices:
            name_lower = (v.name or "").lower()
            # some pyttsx3 voice objects may expose languages
            lang_tags = ",".join(getattr(v, "languages", []) or [])
            if "indonesia" in name_lower or "bahasa" in name_lower or "indonesian" in name_lower:
                chosen = v.id
                break
            if "id" in lang_tags or "ind" in lang_tags:
                chosen = v.id
                break
            # also check id or in voice id string
            vid = (v.id or "").lower()
            if "id_" in vid or vid.endswith("_id") or "indonesia" in vid:
                chosen = v.id
                break
        if chosen:
            engine.setProperty("voice", chosen)
    except Exception:
        pass
    # if no Indonesian voice found, prefer gTTS fallback to ensure Indonesian accent
    if not chosen and gTTS is not None:
        return _tts_with_gtts(text, language)

    engine.save_to_file(text, output_path)
    engine.runAndWait()
    return output_path


def _tts_with_gtts(text: str, language: str = "id") -> str:
    if gTTS is None:
        return _tts_with_wave_fallback(text)

    tmp_dir = tempfile.gettempdir()
    mp3_path = os.path.join(tmp_dir, f"tts_{uuid.uuid4()}.mp3")
    wav_path = os.path.join(tmp_dir, f"tts_{uuid.uuid4()}.wav")

    try:
        tts = gTTS(text, lang=language)
        tts.save(mp3_path)
    except Exception:
        return _tts_with_wave_fallback(text)

    # try to convert mp3 -> wav using ffmpeg if available (PCM16, 44100 Hz mono)
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            mp3_path,
            "-ar",
            "44100",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            wav_path,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                os.remove(mp3_path)
            except Exception:
                pass
            return wav_path
        except Exception:
            return mp3_path
    else:
        return mp3_path


def _tts_with_edge(text: str, language: str = "id") -> str:
    if edge_tts is None:
        return _tts_with_gtts(text, language)

    tmp_dir = tempfile.gettempdir()
    wav_path = os.path.join(tmp_dir, f"tts_{uuid.uuid4()}.wav")

    # choose a default Indonesian neural voice for Edge
    if language and language.startswith("id"):
        default_voice = os.getenv("EDGE_TTS_VOICE", "id-ID-GadisNeural")
    else:
        default_voice = os.getenv("EDGE_TTS_VOICE", "en-US-AriaNeural")

    async def _synth():
        communicate = edge_tts.Communicate(text, default_voice)
        await communicate.save(wav_path)

    try:
        asyncio.run(_synth())
    except Exception:
        return _tts_with_gtts(text, language)

    return wav_path

def transcribe_text_to_speech(text: str, language: str = "id") -> str:
    """
    Fungsi untuk mengonversi teks menjadi suara. Prefer Coqui jika tersedia,
    lalu fallback ke pyttsx3 dan wave generator. `language` membantu memilih
    voice pada fallback.
    """
    lang = os.getenv("TTS_LANGUAGE", language)
    backend = os.getenv("TTS_BACKEND", "auto").lower()

    if backend == "gtts":
        return _tts_with_gtts(text, lang)
    if backend == "pyttsx3":
        return _tts_with_pyttsx3(text, lang)
    if backend == "edge":
        # ensure EDGE_TTS_VOICE favors Indonesian neural voice when language is Indonesian
        if lang and lang.startswith("id"):
            os.environ.setdefault("EDGE_TTS_VOICE", "id-ID-GadisNeural")
        return _tts_with_edge(text, lang)
    if backend == "coqui":
        if _coqui_available():
            return _tts_with_coqui(text)
        # fallback to gTTS then pyttsx3
        if gTTS is not None:
            return _tts_with_gtts(text, lang)
        return _tts_with_pyttsx3(text, lang)
    if backend == "wave":
        return _tts_with_wave_fallback(text)

    # auto: prefer Coqui if available, else prefer pyttsx3 if it seems to have an Indonesian voice,
    # else use gTTS, finally wave fallback
    if _coqui_available():
        return _tts_with_coqui(text)

    if pyttsx3 is not None:
        try:
            engine = _load_pyttsx3_engine()
            voices = engine.getProperty("voices")
            for v in voices:
                name_lower = (v.name or "").lower()
                lang_tags = ",".join(getattr(v, "languages", []) or [])
                vid = (v.id or "").lower()
                if (
                    "indonesia" in name_lower
                    or "bahasa" in name_lower
                    or "indonesian" in name_lower
                    or "id" in lang_tags
                    or "id_" in vid
                    or vid.endswith("_id")
                ):
                    return _tts_with_pyttsx3(text, lang)
        except Exception:
            pass

    if gTTS is not None:
        return _tts_with_gtts(text, lang)

    if pyttsx3 is not None:
        return _tts_with_pyttsx3(text, lang)

    return _tts_with_wave_fallback(text)


def _coqui_available() -> bool:
    model_path = Path(os.getenv("COQUI_MODEL_PATH", COQUI_MODEL_PATH))
    config_path = Path(os.getenv("COQUI_CONFIG_PATH", COQUI_CONFIG_PATH))
    return model_path.exists() and config_path.exists()

# === ENGINE 1: Coqui TTS ===
def _tts_with_coqui(text: str) -> str:
    tmp_dir = tempfile.gettempdir()
    output_path = os.path.join(tmp_dir, f"tts_{uuid.uuid4()}.wav")

    model_path = Path(os.getenv("COQUI_MODEL_PATH", COQUI_MODEL_PATH))
    config_path = Path(os.getenv("COQUI_CONFIG_PATH", COQUI_CONFIG_PATH))
    speaker = os.getenv("COQUI_SPEAKER", COQUI_SPEAKER)

    if not model_path.exists() or not config_path.exists():
        return _tts_with_pyttsx3(text)

    # jalankan Coqui TTS dengan subprocess
    cmd = [
        "tts",
        "--text", text,
        "--model_path", str(model_path),
        "--config_path", str(config_path),
        "--speaker_idx", speaker,
        "--out_path", output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] TTS subprocess failed: {e}")
        return _tts_with_pyttsx3(text)

    return output_path
