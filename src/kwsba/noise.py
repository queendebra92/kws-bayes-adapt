"""Noise sources (babble, environmental, white) and SNR-controlled mixing.

MUSAN files are partitioned *by file* into train / val / test pools (70/10/20,
fixed permutation seed 12345) so that noise used for augmentation or
validation is never reused at test time.

  environmental : a random 1-s segment of a random file in musan/noise
  babble        : sum of K=5 RMS-normalised 1-s segments from distinct files in
                  musan/speech
  white         : Gaussian noise
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .data import L

NOISE_KINDS = ("babble", "environmental", "white")
POOL_SEED = 12345
BABBLE_K = 5


def _partition(files: list[str]) -> dict[str, list[str]]:
    files = sorted(files)
    perm = np.random.default_rng(POOL_SEED).permutation(len(files))
    n_tr, n_va = int(0.7 * len(files)), int(0.1 * len(files))
    idx = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
    return {k: [files[i] for i in v] for k, v in idx.items()}


class NoisePool:
    def __init__(self, musan_root):
        root = Path(musan_root)
        env = [str(p) for p in (root / "noise").rglob("*.wav")]
        spe = [str(p) for p in (root / "speech").rglob("*.wav")]
        if not env or not spe:
            raise FileNotFoundError(f"need musan/noise and musan/speech wavs under {root}")
        self.files = {"environmental": _partition(env), "speech": _partition(spe)}
        self._maps: dict[str, np.ndarray] = {}

    def counts(self) -> dict:
        return {k: {p: len(v) for p, v in d.items()} for k, d in self.files.items()}

    def _sig(self, path: str) -> np.ndarray:
        if path not in self._maps:
            _, x = wavfile.read(path, mmap=True)
            self._maps[path] = x if x.ndim == 1 else x[:, 0]
        return self._maps[path]

    def _segment(self, path: str, rng) -> np.ndarray:
        sig = self._sig(path)
        if len(sig) <= L:
            seg = np.resize(np.asarray(sig), L)
        else:
            start = int(rng.integers(0, len(sig) - L + 1))
            seg = np.asarray(sig[start:start + L])
        return seg.astype(np.float32) / 32768.0

    def sample(self, kind: str, n: int, rng, part: str) -> np.ndarray:
        """Return n noise clips, float32 [n, L]."""
        if kind == "white":
            return rng.standard_normal((n, L)).astype(np.float32)
        out = np.zeros((n, L), dtype=np.float32)
        if kind == "environmental":
            pool = self.files["environmental"][part]
            for i in range(n):
                out[i] = self._segment(pool[int(rng.integers(len(pool)))], rng)
            return out
        if kind == "babble":
            pool = self.files["speech"][part]
            k = min(BABBLE_K, len(pool))
            for i in range(n):
                for j in rng.choice(len(pool), size=k, replace=False):
                    seg = self._segment(pool[int(j)], rng)
                    rms = float(np.sqrt(np.mean(seg ** 2)))
                    out[i] += seg / rms if rms > 1e-6 else seg
            return out
        raise ValueError(f"unknown noise kind {kind!r}")


def to_float(wav_int16: np.ndarray) -> np.ndarray:
    return np.asarray(wav_int16, dtype=np.float32) / 32768.0


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db) -> np.ndarray:
    """clean + g * noise with per-clip g so that 10 log10(Ps / Pn) = snr_db.

    clean, noise: float32 [N, L]; snr_db: scalar or [N]. Power is the mean square
    over the full 1-s clip (zero padding included), floored to avoid /0.
    """
    ps = np.maximum(np.mean(clean.astype(np.float64) ** 2, axis=1), 1e-10)
    pn = np.maximum(np.mean(noise.astype(np.float64) ** 2, axis=1), 1e-10)
    snr = np.broadcast_to(np.asarray(snr_db, dtype=np.float64), ps.shape)
    g = np.sqrt(ps / (pn * 10.0 ** (snr / 10.0)))
    return (clean + g[:, None] * noise).astype(np.float32)


def add_noise(wav_int16, kind: str, snr_db, pool: NoisePool, rng, part: str) -> np.ndarray:
    clean = to_float(wav_int16)
    return mix_at_snr(clean, pool.sample(kind, len(clean), rng, part), snr_db)


def add_random_noise(wav_int16, pool: NoisePool, rng, part: str,
                     snr_range=(0.0, 20.0)) -> np.ndarray:
    """One noisy copy per clip: noise kind and SNR drawn per clip (augmentation)."""
    clean = to_float(wav_int16)
    n = len(clean)
    kinds = rng.integers(len(NOISE_KINDS), size=n)
    snr = rng.uniform(snr_range[0], snr_range[1], size=n)
    out = np.empty_like(clean)
    for k, name in enumerate(NOISE_KINDS):
        m = kinds == k
        if m.any():
            out[m] = mix_at_snr(clean[m], pool.sample(name, int(m.sum()), rng, part), snr[m])
    return out
