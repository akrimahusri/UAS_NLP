from pathlib import Path
import wave
import numpy as np
import sys

p = Path('outputs/tts')
if not p.exists():
    print('no outputs/tts directory')
    sys.exit(0)
files = sorted(p.glob('*.wav'))
if not files:
    print('no wav files in outputs/tts')
    sys.exit(0)

for f in files:
    print('\n---')
    print('file:', f)
    raw = f.read_bytes()
    print('first 16 bytes hex:', raw[:16].hex())
    print('first 3 bytes:', raw[:3])
    try:
        with wave.open(str(f), 'rb') as w:
            nchannels = w.getnchannels()
            sampwidth = w.getsampwidth()
            framerate = w.getframerate()
            nframes = w.getnframes()
            comptype = w.getcomptype()
            compname = w.getcompname()
            print('wave: channels', nchannels, 'sampwidth', sampwidth, 'framerate', framerate, 'nframes', nframes, 'comptype', comptype, 'compname', compname)
            frames = w.readframes(min(nframes, 48000))
            # interpret frames
            if sampwidth == 2:
                dtype = np.int16
            elif sampwidth == 4:
                dtype = np.int32
            elif sampwidth == 1:
                dtype = np.uint8
            else:
                dtype = None
            if dtype is not None:
                arr = np.frombuffer(frames, dtype=dtype)
                if nchannels > 1:
                    arr = arr.reshape(-1, nchannels)
                    mono = arr.mean(axis=1)
                else:
                    mono = arr
                mono = mono.astype(np.float64)
                # normalize based on dtype
                if sampwidth == 2:
                    mono = mono / 32768.0
                elif sampwidth == 4:
                    mono = mono / 2147483648.0
                elif sampwidth == 1:
                    mono = (mono - 128) / 128.0
                print('samples:', 'min', float(mono.min()), 'max', float(mono.max()), 'mean', float(mono.mean()))
                rms = np.sqrt(np.mean(mono**2))
                print('rms:', float(rms))
                zeros = np.mean(np.isclose(mono, 0.0))
                print('fraction zeros:', float(zeros))
            else:
                print('unknown sampwidth, cannot parse frames')
    except wave.Error as e:
        print('wave open error:', e)
    # quick check if file looks like mp3 data
    if raw[:3] == b'ID3' or raw[:2] == b'\xff\xfb':
        print('looks like mp3 data')
    else:
        print('not mp3 by header')
