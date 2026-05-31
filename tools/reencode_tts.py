from pathlib import Path
import shutil
import subprocess

p = Path('outputs/tts')
if not p.exists():
    print('no outputs/tts')
    raise SystemExit(1)

wav_files = sorted(p.glob('*.wav'))
if not wav_files:
    print('no wav files')
    raise SystemExit(0)

for f in wav_files:
    out = f.with_name(f.stem + '_conv44100.wav')
    # skip if already 44100
    try:
        import wave
        with wave.open(str(f),'rb') as w:
            if w.getframerate() == 44100 and w.getsampwidth() == 2 and w.getnchannels() == 1:
                print('skip (already good):', f.name)
                continue
    except Exception:
        pass

    if not shutil.which('ffmpeg'):
        print('ffmpeg not found; skipping re-encode')
        break

    cmd = [
        'ffmpeg','-y','-i',str(f),'-ar','44100','-ac','1','-c:a','pcm_s16le',str(out)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print('rewrote:', f.name, '->', out.name)
    except Exception as e:
        print('failed:', f.name, e)
