#!/usr/bin/env python
"""Write a multichannel wav containing one far-field source.

Exact fractional delays are applied in the frequency domain, so the file can
be used to sanity-check a model's front end (SRP map, GCC, ...) without a
simulator.  Elevation is zenith (90 deg = horizontal).

Example::

    python examples/make_synthetic.py \
      --meta /path/to/dataset_meta.json --mic-indices 5,6,8 \
      --az 60 --el 90 --duration 0.333 --out /tmp/source_az60.wav
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True,
                   help="dataset_meta.json containing mic_pos")
    p.add_argument("--mic-indices", default="5,6,8")
    p.add_argument("--az", type=float, default=60.0)
    p.add_argument("--el", type=float, default=90.0,
                   help="zenith elevation (90 = horizontal)")
    p.add_argument("--duration", type=float, default=16000 / 48000)
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--snr-db", type=float, default=None,
                   help="optional white-noise SNR")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    meta = json.loads(Path(args.meta).read_text())
    indices = [int(v) for v in args.mic_indices.split(",")]
    mic = np.asarray(meta["mic_pos"], dtype=np.float64)[indices]
    mic = mic - mic.mean(axis=0, keepdims=True)

    fs = args.sample_rate
    n = int(round(args.duration * fs))
    rng = np.random.default_rng(args.seed)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)

    az, el = np.deg2rad(args.az), np.deg2rad(args.el)
    u = np.array([np.sin(el) * np.cos(az), np.sin(el) * np.sin(az), np.cos(el)])
    channels = []
    for pos in mic:
        delay = float(np.dot(pos, u)) / 343.0
        channels.append(np.fft.irfft(spectrum * np.exp(2j * np.pi * freqs * delay), n=n))
    audio = np.stack(channels, axis=-1)
    audio /= np.max(np.abs(audio)) + 1e-12

    if args.snr_db is not None:
        noise = rng.standard_normal(audio.shape)
        noise *= np.sqrt(np.mean(audio ** 2) / (10 ** (args.snr_db / 10.0)))
        audio = audio + noise
        audio /= np.max(np.abs(audio)) + 1e-12

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, audio.astype(np.float32), fs)
    print(f"wrote {args.out}  shape={audio.shape} fs={fs} "
          f"az={args.az} el(zenith)={args.el}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
