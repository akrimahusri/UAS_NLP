from __future__ import annotations

import argparse
from pathlib import Path

import os
# prevent Transformers from loading TensorFlow (avoid protobuf runtime conflicts)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.pipeline import OUTPUT_DIR, process_audio_bytes, process_dataset


app = FastAPI(title="Voice Chatbot UAS", version="1.0.0")


@app.get("/health")
def health_check():
	return {"status": "ok"}


@app.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...), mode: str = "preserve_cs"):
	file_bytes = await file.read()
	if not file_bytes:
		raise HTTPException(status_code=400, detail="File audio kosong")

	result = process_audio_bytes(file_bytes, file.filename or "input.wav", response_mode=mode)
	if result.status != "ok" or not result.audio_output:
		raise HTTPException(status_code=500, detail=result.error or "Pipeline gagal")

	# determine media type from file extension to serve correct audio mime
	suffix = Path(result.audio_output).suffix.lower()
	if suffix in {".mp3", ".mpeg"}:
		media_type = "audio/mpeg"
	elif suffix in {".wav"}:
		media_type = "audio/wav"
	else:
		media_type = "application/octet-stream"

	return FileResponse(
		result.audio_output,
		media_type=media_type,
		filename=Path(result.audio_output).name,
	)


@app.post("/process-file")
async def process_file(file: UploadFile = File(...), mode: str = "preserve_cs"):
	file_bytes = await file.read()
	if not file_bytes:
		raise HTTPException(status_code=400, detail="File audio kosong")

	result = process_audio_bytes(file_bytes, file.filename or "input.wav", response_mode=mode)
	return JSONResponse(result.__dict__)


@app.post("/process-dataset")
def process_dataset_endpoint(limit: int | None = None):
	summary = process_dataset(limit=limit)
	return JSONResponse(summary)


def _cli() -> None:
	parser = argparse.ArgumentParser(description="Voice chatbot backend and dataset processor")
	parser.add_argument("--dataset", type=str, default=None, help="Folder dataset audio untuk diproses")
	parser.add_argument("--limit", type=int, default=None, help="Batas jumlah file audio yang diproses")
	parser.add_argument(
		"--mode",
		type=str,
		default="preserve_cs",
		choices=["preserve_cs", "normalized"],
		help="Mode respons LLM",
	)
	parser.add_argument(
		"--tts-backend",
		type=str,
		choices=["auto", "coqui", "gtts", "pyttsx3", "edge", "wave"],
		default=None,
		help="Pilih backend TTS (env TTS_BACKEND).",
	)
	parser.add_argument(
		"--progress",
		action="store_true",
		help="Tampilkan progres pemrosesan dataset per file",
	)
	parser.add_argument(
		"--resume",
		action="store_true",
		help="Lanjutkan dari run sebelumnya, lewati file yang sudah diproses",
	)
	parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Folder output eksperimen")
	args = parser.parse_args()

	if args.dataset:
		if args.tts_backend:
			import os
			os.environ["TTS_BACKEND"] = args.tts_backend
		summary = process_dataset(
			args.dataset,
			args.output,
			args.limit,
			response_mode=args.mode,
			show_progress=args.progress,
			resume=args.resume,
		)
		print(summary)
		return

	import uvicorn

	uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
	_cli()

