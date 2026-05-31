import traceback

try:
    from faster_whisper import WhisperModel
    print('import_ok')
except Exception:
    traceback.print_exc()
