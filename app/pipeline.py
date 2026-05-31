from __future__ import annotations

import json
import csv
import contextlib
import re
import wave
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import copy2
import shutil
import subprocess
from typing import Iterable

from app.llm import ApiQuotaExceededError, generate_response
from app.stt import transcribe_speech_to_text
from app.tts import transcribe_text_to_speech
from app.normalize import normalize_and_tag


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FILENAME_PATTERN = re.compile(r"^(?P<speaker>\d{4})_(?P<utterance>[A-Za-z0-9]+)\.wav$", re.IGNORECASE)


@dataclass(slots=True)
class PipelineResult:
    source_file: str
    response_mode: str
    transcript: str
    normalized_text: str | None
    tagged_input: str | None
    response_text: str
    audio_output: str | None
    status: str
    processing_seconds: float
    error: str | None = None


def process_audio_bytes(
    file_bytes: bytes,
    file_name: str = "input.wav",
    response_mode: str = "preserve_cs",
) -> PipelineResult:
    started_at = time.perf_counter()
    transcript = transcribe_speech_to_text(file_bytes, Path(file_name).suffix or ".wav")

    if transcript.startswith("[ERROR]"):
        return PipelineResult(
            source_file=file_name,
            response_mode=response_mode,
            transcript=transcript,
            normalized_text=None,
            tagged_input=None,
            response_text="",
            audio_output=None,
            status="stt_failed",
            processing_seconds=time.perf_counter() - started_at,
            error=transcript,
        )

    # normalisasi + tagging code-switching sebelum dikirim ke LLM
    normalized = normalize_and_tag(transcript)
    normalized_text = normalized.get("original", transcript)
    tagged_input = normalized.get("tagged", transcript)

    if response_mode == "normalized":
        llm_input = normalized_text
    else:
        llm_input = tagged_input

    response_text = generate_response(llm_input, response_mode=response_mode)
    audio_output = transcribe_text_to_speech(response_text)

    if isinstance(audio_output, str) and audio_output.startswith("[ERROR]"):
        return PipelineResult(
            source_file=file_name,
            response_mode=response_mode,
            transcript=transcript,
            normalized_text=normalized_text,
            tagged_input=tagged_input,
            response_text=response_text,
            audio_output=None,
            status="tts_failed",
            processing_seconds=time.perf_counter() - started_at,
            error=audio_output,
        )

    return PipelineResult(
        source_file=file_name,
        response_mode=response_mode,
        transcript=transcript,
        normalized_text=normalized_text,
        tagged_input=tagged_input,
        response_text=response_text,
        audio_output=audio_output,
        status="ok",
        processing_seconds=time.perf_counter() - started_at,
    )


def process_audio_file(audio_path: str | Path) -> PipelineResult:
    source_path = Path(audio_path)
    return process_audio_bytes(source_path.read_bytes(), source_path.name)


def iter_audio_files(data_dir: str | Path = DATA_DIR) -> Iterable[Path]:
    base_dir = Path(data_dir)
    if not base_dir.exists():
        return []
    return sorted(path for path in base_dir.rglob("*.wav") if path.is_file())


def _dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _audio_duration_seconds(audio_path: Path) -> float | None:
    try:
        with contextlib.closing(wave.open(str(audio_path), "rb")) as wav_file:
            frame_count = wav_file.getnframes()
            frame_rate = wav_file.getframerate() or 1
            return frame_count / float(frame_rate)
    except Exception:
        return None


def _parse_audio_filename(audio_path: Path) -> dict[str, str | None]:
    match = FILENAME_PATTERN.match(audio_path.name)
    if not match:
        return {
            "speaker_id": None,
            "utterance_id": None,
            "filename_valid": False,
        }

    return {
        "speaker_id": match.group("speaker"),
        "utterance_id": match.group("utterance"),
        "filename_valid": True,
    }


def summarize_dataset(data_dir: str | Path = DATA_DIR) -> dict:
    files = list(iter_audio_files(data_dir))
    speaker_counts: dict[str, int] = {}
    total_duration = 0.0
    counted_duration = 0
    filename_issues: list[str] = []

    for audio_file in files:
        parsed = _parse_audio_filename(audio_file)
        speaker_id = parsed["speaker_id"] or audio_file.stem.split("_")[0]
        speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
        if not parsed["filename_valid"]:
            filename_issues.append(audio_file.name)
        duration = _audio_duration_seconds(audio_file)
        if duration is not None:
            total_duration += duration
            counted_duration += 1

    speakers_below_minimum = {
        speaker_id: count for speaker_id, count in speaker_counts.items() if count < 10
    }

    return {
        "data_dir": str(Path(data_dir).resolve()),
        "total_files": len(files),
        "unique_speakers": len(speaker_counts),
        "speaker_counts": speaker_counts,
        "filename_issues": filename_issues,
        "filename_issue_count": len(filename_issues),
        "speakers_below_minimum_10": speakers_below_minimum,
        "total_duration_seconds": round(total_duration, 2),
        "average_duration_seconds": round(total_duration / counted_duration, 2) if counted_duration else None,
        "duration_counted_files": counted_duration,
    }


def process_dataset(
    data_dir: str | Path = DATA_DIR,
    output_dir: str | Path = OUTPUT_DIR,
    limit: int | None = None,
    response_mode: str = "preserve_cs",
    show_progress: bool = False,
    resume: bool = False,
) -> dict:
    files = list(iter_audio_files(data_dir))
    if limit is not None:
        files = files[: max(limit, 0)]

    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    corpus_summary = summarize_dataset(data_dir)

    # if resume requested, try to load existing results so we don't re-run files
    results: list[dict] = []
    processed_files: set[str] = set()
    existing_jsonl = output_base / "experiment_results.jsonl"
    if resume and existing_jsonl.exists():
        try:
            for line in existing_jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    results.append(obj)
                    # consider response_mode as well so resume is mode-specific
                    key = f"{obj.get('source_file','')}||{obj.get('response_mode','') }"
                    processed_files.add(key)
                except Exception:
                    # skip malformed lines
                    continue
        except Exception:
            # ignore resume load errors and start fresh
            results = []
            processed_files = set()
    total_files = len(files)
    stopped_due_to_api_limit = False
    stop_error_message: str | None = None
    for index, audio_file in enumerate(files, start=1):
        # skip if this exact file+mode already processed
        composite_key = f"{audio_file.name}||{response_mode}"
        if composite_key in processed_files:
            if show_progress:
                print(f"[{index}/{total_files}] Skipping already-processed {audio_file.name} (mode={response_mode})")
            continue
        if show_progress:
            print(f"[{index}/{total_files}] Processing {audio_file.name}...")
        try:
            result = process_audio_bytes(audio_file.read_bytes(), audio_file.name, response_mode=response_mode)
        except ApiQuotaExceededError as exc:
            stopped_due_to_api_limit = True
            stop_error_message = str(exc)
            if show_progress:
                print(f"[STOP] API limit terdeteksi saat memproses {audio_file.name}: {exc}")
            break
        result_dict = asdict(result)
        parsed = _parse_audio_filename(audio_file)
        result_dict["speaker_id"] = parsed["speaker_id"] or audio_file.stem.split("_")[0]
        result_dict["utterance_id"] = parsed["utterance_id"]
        result_dict["filename_valid"] = parsed["filename_valid"]
        # copy TTS output to outputs/tts with distinct name to avoid confusion
        if result.audio_output and Path(result.audio_output).exists():
            target_dir = output_base / "tts"
            target_dir.mkdir(parents=True, exist_ok=True)
            src_path = Path(result.audio_output)
            src_suffix = src_path.suffix.lower() or ".wav"

            # If source is mp3 and ffmpeg exists, convert to WAV PCM16 to avoid playback issues
            if src_suffix in {".mp3", ".mpeg"} and shutil.which("ffmpeg"):
                safe_name = f"{audio_file.stem}_resp_{response_mode}.wav"
                target_path = target_dir / safe_name
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(src_path),
                    "-ar",
                    "44100",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(target_path),
                ]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    result_dict["audio_output"] = str(target_path)
                except Exception:
                    # fallback: copy original mp3
                    try:
                        target_path = target_dir / f"{audio_file.stem}_resp_{response_mode}{src_suffix}"
                        copy2(src_path, target_path)
                        result_dict["audio_output"] = str(target_path)
                    except Exception:
                        result_dict["audio_output"] = result_dict.get("audio_output")
            else:
                # normalize any audio to WAV PCM16 44100 mono when possible to avoid
                # playback/resampling artifacts. If ffmpeg is available, re-encode
                # everything to a consistent WAV. Otherwise preserve original.
                safe_name = f"{audio_file.stem}_resp_{response_mode}.wav"
                target_path = target_dir / safe_name
                if shutil.which("ffmpeg"):
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(src_path),
                        "-ar",
                        "44100",
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_s16le",
                        str(target_path),
                    ]
                    try:
                        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        result_dict["audio_output"] = str(target_path)
                    except Exception:
                        try:
                            # fallback: copy original (preserve extension)
                            target_path = target_dir / f"{audio_file.stem}_resp_{response_mode}{src_suffix}"
                            copy2(src_path, target_path)
                            result_dict["audio_output"] = str(target_path)
                        except Exception:
                            result_dict["audio_output"] = result_dict.get("audio_output")
                else:
                    try:
                        copy2(src_path, target_path)
                        result_dict["audio_output"] = str(target_path)
                    except Exception:
                        result_dict["audio_output"] = result_dict.get("audio_output")

        results.append(result_dict)

        # append this record immediately to JSONL so we can monitor progress live
        try:
            with open(output_base / "experiment_results.jsonl", "a", encoding="utf-8") as jl:
                jl.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
        except Exception:
            pass

        # update a lightweight progress summary file
        try:
            partial = {
                "data_dir": str(Path(data_dir).resolve()),
                "processed_files": len(results),
                "total_to_process": total_files,
                "last_processed": result_dict.get("source_file"),
                "successful": sum(1 for item in results if item.get("status") == "ok"),
                "stt_failed": sum(1 for item in results if item.get("status") == "stt_failed"),
                "tts_failed": sum(1 for item in results if item.get("status") == "tts_failed"),
            }
            with open(output_base / "experiment_progress.json", "w", encoding="utf-8") as pf:
                pf.write(json.dumps(partial, ensure_ascii=False, indent=2))
        except Exception:
            pass

        if show_progress:
            elapsed = f"{result.processing_seconds:.2f}s"
            print(f"[{index}/{total_files}] Done {audio_file.name} ({elapsed}, status={result.status})")

        # copying already handled above before appending

    summary = {
        "data_dir": str(Path(data_dir).resolve()),
        "total_files": len(files),
        "successful": sum(1 for item in results if item["status"] == "ok"),
        "stt_failed": sum(1 for item in results if item["status"] == "stt_failed"),
        "tts_failed": sum(1 for item in results if item["status"] == "tts_failed"),
        "response_mode": response_mode,
        "stopped_due_to_api_limit": stopped_due_to_api_limit,
        "stop_error_message": stop_error_message,
        "corpus_summary": corpus_summary,
        "results": results,
        "evaluation_notes": {
            "wer": None,
            "cer": None,
            "wer_cer_reason": "Belum ada transkrip referensi di folder Data, jadi WER/CER belum bisa dihitung otomatis.",
            "manual_llm_tts_review": "Disarankan menilai kualitas jawaban LLM dan naturalness TTS secara manual pada sampel hasil batch.",
        },
    }

    _dump_json(output_base / "experiment_results.json", summary)
    (output_base / "experiment_results.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in results),
        encoding="utf-8",
    )

    # export CSV for easier spreadsheet analysis
    csv_path = output_base / "experiment_results.csv"
    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            header = [
                "source_file",
                "speaker_id",
                "utterance_id",
                "filename_valid",
                "response_mode",
                "status",
                "processing_seconds",
                "transcript",
                "normalized_text",
                "tagged_input",
                "response_text",
                "audio_output",
                "error",
            ]
            writer.writerow(header)
            for item in results:
                writer.writerow(
                    [
                        item.get("source_file"),
                        item.get("speaker_id"),
                        item.get("utterance_id"),
                        item.get("filename_valid"),
                        item.get("response_mode"),
                        item.get("status"),
                        item.get("processing_seconds"),
                        (item.get("transcript") or "").replace("\n", " "),
                        (item.get("normalized_text") or "").replace("\n", " "),
                        (item.get("tagged_input") or "").replace("\n", " "),
                        (item.get("response_text") or "").replace("\n", " "),
                        item.get("audio_output"),
                        (item.get("error") or ""),
                    ]
                )
    except Exception:
        # best-effort CSV export; ignore failures
        pass
    _dump_json(output_base / "dataset_summary.json", corpus_summary)

    markdown_lines = [
        "# Experiment Results",
        "",
        f"- Total files: {summary['total_files']}",
        f"- Successful: {summary['successful']}",
        f"- STT failed: {summary['stt_failed']}",
        f"- TTS failed: {summary['tts_failed']}",
        f"- Response mode: {summary['response_mode']}",
        f"- Stopped due to API limit: {summary['stopped_due_to_api_limit']}",
        f"- Unique speakers: {corpus_summary['unique_speakers']}",
        f"- Total duration (sec): {corpus_summary['total_duration_seconds']}",
        f"- Filename issues: {corpus_summary['filename_issue_count']}",
        "",
        "## Corpus Audit",
        f"- Speakers below minimum 10 recordings: {len(corpus_summary['speakers_below_minimum_10'])}",
        f"- WER/CER: unavailable without reference transcripts",
        "",
        "## Sample rows",
    ]
    for item in results[:10]:
        markdown_lines.append(
            f"- {item['source_file']}: {item['status']} | {item['transcript'][:120]} | {item['response_text'][:120]}"
        )

    (output_base / "experiment_results.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    return summary
