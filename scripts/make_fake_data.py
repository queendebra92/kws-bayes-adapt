"""Generate a tiny SYNTHETIC stand-in for Speech Commands + MUSAN.

For pipeline smoke tests only. The audio is synthetic tones, so any metric
computed on it says nothing about real keyword spotting.

    python scripts/make_fake_data.py fake_data
"""
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

SR = 16000
KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
OTHER = ["bed", "bird", "cat", "dog", "happy"]


def tone_clip(freqs, rng, dur=1.0):
    t = np.arange(int(SR * dur)) / SR
    x = sum(np.sin(2 * np.pi * f * (1 + 0.02 * rng.standard_normal()) * t + rng.uniform(0, 6.28))
            for f in freqs)
    env = np.hanning(len(t)) ** 0.5
    x = x * env * rng.uniform(0.05, 0.15) + 0.002 * rng.standard_normal(len(t))
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def main(root):
    root = Path(root)
    sc, mu = root / "speech_commands", root / "musan"
    rng = np.random.default_rng(0)
    freq = {w: [300 + 90 * i, 900 + 140 * ((3 * i) % 7), 2200 + 200 * ((5 * i) % 5)]
            for i, w in enumerate(KEYWORDS + OTHER)}
    val, test = [], []
    for w in KEYWORDS + OTHER:
        (sc / w).mkdir(parents=True, exist_ok=True)
        for k in range(40):
            rel = f"{w}/spk{k:04d}_nohash_0.wav"
            wavfile.write(sc / rel, SR, tone_clip(freq[w], rng))
            if k % 10 == 0:
                val.append(rel)
            elif k % 10 == 1:
                test.append(rel)
    (sc / "validation_list.txt").write_text("\n".join(val) + "\n")
    (sc / "testing_list.txt").write_text("\n".join(test) + "\n")
    (sc / "_background_noise_").mkdir(exist_ok=True)
    for i in range(6):
        x = (0.05 * rng.standard_normal(SR * 40)).astype(np.float32)
        wavfile.write(sc / "_background_noise_" / f"bg{i}.wav", SR, (x * 32767).astype(np.int16))
    for sub, n, dur in (("noise", 20, 12), ("speech", 20, 12)):
        (mu / sub).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            x = 0.05 * rng.standard_normal(SR * dur)
            if sub == "speech":
                x = x * (1 + np.sin(2 * np.pi * 4 * np.arange(len(x)) / SR))
            wavfile.write(mu / sub / f"{sub}{i:03d}.wav", SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    print("fake data in", root)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fake_data")
