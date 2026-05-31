# Voice Chatbot UAS – STT, Gemini LLM, TTS Integration

Proyek UAS ini adalah aplikasi end-to-end voice chatbot (STT → normalize/tag → LLM → TTS). Tujuan: eksperimen pipeline yang mempertahankan code-switching, menyediakan mode respons terkontrol, menulis artifact eksperimen, dan mendukung fallback multi-backend untuk STT/LLM/TTS.

## 📌 Fitur Utama
- 🎙️ Speech-to-Text (STT) menggunakan `whisper.cpp` dari OpenAI atau fallback Whisper lokal.
- 🧠 LLM Integration menggunakan Google Gemini API untuk menghasilkan respons dalam Bahasa Indonesia.
- 🔀 Processing layer opsional untuk normalisasi transkrip, deteksi bahasa, dan penandaan code-switching.
- 🔊 Text-to-Speech (TTS) menggunakan model Coqui TTS (Indonesian TTS) atau fallback lokal.
- 🧪 Antarmuka pengguna interaktif berbasis `Gradio` untuk pengujian langsung dari browser.

## 🗂️ Struktur Proyek
```
voice_chatbot_project/
│
├── app/                   # Core backend: main, llm, stt, tts, normalize
├── gradio_app/            # Optional demo UI with Gradio
├── tools/                 # Helper scripts (analysis, re-encode, regenerate)
├── Data/                  # (dataset audio) - DO NOT push large datasets
├── outputs/               # Experiment artifacts and TTS results
├── requirements.txt
├── README.md
```

## ▶️ Cara Menjalankan
1. Siapkan virtual environment (direkomendasikan) dan install dependensi:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Jalankan backend FastAPI (development):

```powershell
.venv\Scripts\python -m app.main
```

3. Jalankan demo Gradio (opsional):

```powershell
.venv\Scripts\python gradio_app/app.py
```

4. Jalankan eksperimen batch (contoh):

```powershell
.venv\Scripts\python -m app.main --dataset Data --mode preserve_cs --progress
```

5. Opsi penting:
- Resume run (skip file yg sudah tercatat di `outputs/experiment_results.jsonl`):

```powershell
.venv\Scripts\python -m app.main --dataset Data --mode preserve_cs --progress --resume
```

- Pilih backend TTS (sekarang mendukung `edge`):

```powershell
.venv\Scripts\python -m app.main --dataset Data --progress --tts-backend edge
```

Catatan: jalankan dengan `.venv\Scripts\python` supaya paket yang terpasang di virtualenv (mis. `faster-whisper`, `edge-tts`) digunakan.

## 📦 Output Eksperimen
	- `outputs/experiment_results.jsonl` (append per-file)
	- `outputs/experiment_results.json` (ringkasan)
	- `outputs/experiment_results.csv` (eksport spreadsheet)
	- `outputs/dataset_summary.json`
	- `outputs/experiment_progress.json`

## 👨‍💻 Dibuat Untuk
Proyek UAS mata kuliah *Pemrosesan Bahasa Alami* — Semester Genap 2024/2025.
