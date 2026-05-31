from pathlib import Path
import wave
p = Path('outputs/tts')
if not p.exists():
    print('no tts dir')
    raise SystemExit(0)
files = sorted(p.glob('*'))
print('found', len(files), 'files')
for f in files[:10]:
    print('---', f.name)
    b = f.read_bytes()[:64]
    print('prefix bytes:', list(b[:12]))
    sfx = f.suffix.lower()
    print('suffix', sfx)
    if sfx == '.wav':
        try:
            with wave.open(str(f), 'rb') as w:
                print('wave: channels', w.getnchannels(), 'sampwidth', w.getsampwidth(), 'framerate', w.getframerate(), 'nframes', w.getnframes())
        except Exception as e:
            print('wave open error:', e)
    else:
        if b[:3] == b'ID3' or (len(b) > 0 and b[0] == 0xFF):
            print('looks like mp3 data')
        else:
            print('unknown header')
print('done')
