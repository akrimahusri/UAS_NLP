#!/usr/bin/env python3
"""Rekam audio dari mikrofon dan simpan sebagai WAV 44100 Hz mono PCM16.

Usage:
  .venv\Scripts\python tools\record_mic.py --outfile sample.wav --seconds 5

Instal dependensi jika belum:
  pip install sounddevice soundfile
"""
from __future__ import annotations

import argparse
import sounddevice as sd
import soundfile as sf


def record(outfile: str, seconds: float, samplerate: int = 44100, channels: int = 1):
    print(f"Recording {seconds}s -> {outfile} @ {samplerate}Hz {channels}ch")
    data = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
    sd.wait()
    sf.write(outfile, data, samplerate, subtype='PCM_16')
    print("Done")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outfile", required=True, help="Output WAV file path")
    p.add_argument("--seconds", type=float, default=5.0, help="Duration in seconds")
    p.add_argument("--samplerate", type=int, default=44100)
    p.add_argument("--channels", type=int, default=1)
    args = p.parse_args()
    record(args.outfile, args.seconds, args.samplerate, args.channels)


if __name__ == "__main__":
    main()
