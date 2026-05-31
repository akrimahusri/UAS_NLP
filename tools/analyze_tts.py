import wave
from pathlib import Path
import struct
import math
p=Path('outputs/tts')
files=sorted(p.glob('*.wav'))
if not files:
    print('no wav files')
    raise SystemExit
for f in files[:5]:
    print('---', f.name)
    with wave.open(str(f),'rb') as w:
        n = w.getnframes(); fr = w.getframerate(); ch = w.getnchannels(); sw = w.getsampwidth()
        frames = w.readframes(min(n, 200000))
    # unpack as signed 16-bit
    fmt = '<' + 'h'*(len(frames)//2)
    samples = struct.unpack(fmt, frames)
    # compute stats
    import statistics
    rms = math.sqrt(sum(s*s for s in samples)/len(samples))
    mx = max(abs(s) for s in samples)
    print('frames', n, 'framerate', fr, 'channels', ch, 'sampwidth', sw)
    print('rms', round(rms,2), 'max', mx)
    # print first 40 samples
    print('first40', samples[:40])
    # check repeated pattern frequency
    # look for many zeros
    zeros = sum(1 for s in samples if s==0)
    print('zeros', zeros, 'zero_ratio', round(zeros/len(samples),4))
print('done')
