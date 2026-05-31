import os
import uuid
import tempfile
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# path ke folder utilitas STT
WHISPER_DIR = os.path.join(BASE_DIR, "whisper.cpp")

# TODO: Lengkapi path ke binary whisper-cli
# Gunakan os.path.join() untuk menggabungkan WHISPER_DIR, "build", "bin", dan "whisper-cli"
WHISPER_BINARY = os.path.join(WHISPER_DIR, "build", "bin", "whisper-cli.exe")

# TODO: Lengkapi path ke file model Whisper (contoh: ggml-large-v3-turbo.bin)
# Gunakan os.path.join() untuk mengarah ke file model di dalam folder "models"
WHISPER_MODEL_PATH = os.path.join(WHISPER_DIR, "models", "ggml-large-v3-turbo.bin")


@lru_cache(maxsize=1)
def _load_transformers_asr():
    # prevent transformers from importing TensorFlow (avoids protobuf/tf runtime conflicts)
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    from transformers import pipeline

    model_name = os.getenv("WHISPER_HF_MODEL", "openai/whisper-tiny")
    return pipeline("automatic-speech-recognition", model=model_name, device=-1)


def _fallback_whisper_transcribe(audio_path: str) -> str:
    # Try faster-whisper first (local PyTorch-backed whisper implementation)
    try:
        from faster_whisper import WhisperModel

        model_name = os.getenv("WHISPER_HF_MODEL", "openai/whisper-small")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
        segments, info = model.transcribe(audio_path, beam_size=5)
        text = "".join([s.text for s in segments])
        return text.strip()
    except Exception:
        # faster-whisper not available or failed — fall back to transformers
        pass

    asr = _load_transformers_asr()
    sample_rate, samples = wavfile.read(audio_path)
    if isinstance(samples, np.ndarray) and samples.ndim > 1:
        samples = samples.mean(axis=1)
    if np.issubdtype(samples.dtype, np.integer):
        max_value = np.iinfo(samples.dtype).max or 1
        samples = samples.astype(np.float32) / float(max_value)
    else:
        samples = samples.astype(np.float32)

    target_rate = 16000
    if sample_rate != target_rate and sample_rate > 0:
        gcd = np.gcd(sample_rate, target_rate)
        up = target_rate // gcd
        down = sample_rate // gcd
        samples = resample_poly(samples, up, down).astype(np.float32)
        sample_rate = target_rate

    result = asr({"array": samples, "sampling_rate": sample_rate})
    return str(result.get("text", "")).strip()

def transcribe_speech_to_text(file_bytes: bytes, file_ext: str = ".wav") -> str:
    """
    Transkrip file audio menggunakan whisper.cpp CLI
    Args:
        file_bytes (bytes): Isi file audio
        file_ext (str): Ekstensi file, default ".wav"
    Returns:
        str: Teks hasil transkripsi
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, f"{uuid.uuid4()}{file_ext}")
        result_base = os.path.join(tmpdir, "transcription")

        # simpan audio ke file temporer
        with open(audio_path, "wb") as f:
            f.write(file_bytes)

        whisper_binary = Path(os.getenv("WHISPER_BINARY", WHISPER_BINARY))
        whisper_model = Path(os.getenv("WHISPER_MODEL_PATH", WHISPER_MODEL_PATH))

        if whisper_binary.exists() and whisper_model.exists():
            # jalankan whisper.cpp dengan subprocess
            cmd = [
                str(whisper_binary),
                "-m", str(whisper_model),
                "-f", audio_path,
                "-otxt",
                "-of", result_base,
            ]

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                return f"[ERROR] Whisper failed: {e}"

            result_path = f"{result_base}.txt"
            try:
                with open(result_path, "r", encoding="utf-8") as result_file:
                    return result_file.read().strip()
            except FileNotFoundError:
                return "[ERROR] Transcription file not found"

        try:
            return _fallback_whisper_transcribe(audio_path)
        except Exception as exc:
            return f"[ERROR] Whisper fallback failed: {exc}"
