"""MFCC features in plain NumPy/SciPy (no librosa needed at run time).

Frame-level:  mfcc(x) -> [N, n_mfcc, T]   (T = 98: 30 ms window, 10 ms hop)
Clip-level:   pooled_features(x) -> [N, 78]
              13 MFCC + delta + delta-delta (39 per frame), mean and std over time.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dct, rfft

from .data import L, SR

WIN, HOP, NFFT = 480, 160, 512
N_MELS, FMIN, FMAX = 40, 20.0, 7600.0
PREEMPH = 0.97
N_FRAMES = 1 + (L - WIN) // HOP  # 98


def _hz2mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel2hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    """Triangular HTK-mel filterbank, shape [N_MELS, NFFT//2+1]."""
    pts = _mel2hz(np.linspace(_hz2mel(FMIN), _hz2mel(FMAX), N_MELS + 2))
    freqs = np.linspace(0.0, SR / 2.0, NFFT // 2 + 1)
    fb = np.zeros((N_MELS, len(freqs)))
    for i in range(N_MELS):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / (c - lo)
        down = (hi - freqs) / (hi - c)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb.astype(np.float32)


_FB = mel_filterbank()
_WINDOW = np.hamming(WIN).astype(np.float32)


def log_mel(x: np.ndarray, batch: int = 512) -> np.ndarray:
    """x float32 [N, L] -> log-mel energies [N, T, N_MELS]."""
    out = np.empty((len(x), N_FRAMES, N_MELS), dtype=np.float32)
    for s in range(0, len(x), batch):
        b = np.asarray(x[s:s + batch], dtype=np.float32)
        b = np.concatenate([b[:, :1], b[:, 1:] - PREEMPH * b[:, :-1]], axis=1)
        frames = np.lib.stride_tricks.sliding_window_view(b, WIN, axis=1)[:, ::HOP][:, :N_FRAMES]
        spec = np.abs(rfft(frames * _WINDOW, n=NFFT, axis=-1)) ** 2
        out[s:s + batch] = np.log(spec @ _FB.T + 1e-10)
    return out


def mfcc(x: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
    """x float32 [N, L] -> MFCC [N, n_mfcc, T]."""
    lm = log_mel(x)
    c = dct(lm, type=2, norm="ortho", axis=-1)[..., :n_mfcc]
    return np.ascontiguousarray(c.transpose(0, 2, 1)).astype(np.float32)


def deltas(c: np.ndarray, width: int = 2) -> np.ndarray:
    """Regression deltas along the last (time) axis; edges replicated."""
    pad = np.pad(c, [(0, 0)] * (c.ndim - 1) + [(width, width)], mode="edge")
    num = np.zeros_like(c)
    for n in range(1, width + 1):
        num += n * (pad[..., width + n:pad.shape[-1] - width + n] -
                    pad[..., width - n:pad.shape[-1] - width - n])
    return num / (2.0 * sum(n * n for n in range(1, width + 1)))


def pooled_features(x: np.ndarray, n_seg: int = 1) -> np.ndarray:
    """x float32 [N, L] -> clip-level vector from 13 MFCC + delta + delta-delta (39 / frame).

    n_seg = 1 : mean and std over the whole clip                      -> 78-d
    n_seg > 1 : mean of each of n_seg equal time segments (keeps coarse
                temporal order) plus the whole-clip std               -> 39*(n_seg+1)-d
    """
    c = mfcc(x, 13)
    d1 = deltas(c)
    d2 = deltas(d1)
    f = np.concatenate([c, d1, d2], axis=1)  # [N, 39, T]
    if n_seg == 1:
        return np.concatenate([f.mean(axis=2), f.std(axis=2)], axis=1).astype(np.float32)
    segs = [f[:, :, idx].mean(axis=2) for idx in np.array_split(np.arange(f.shape[2]), n_seg)]
    return np.concatenate(segs + [f.std(axis=2)], axis=1).astype(np.float32)


def pooled_multi(x: np.ndarray, segs=(1, 4)) -> dict:
    """Compute MFCC once, return {n_seg: features} for several segment counts."""
    c = mfcc(x, 13)
    d1 = deltas(c)
    f = np.concatenate([c, d1, deltas(d1)], axis=1)
    std = f.std(axis=2)
    out = {}
    for n in segs:
        if n == 1:
            out[n] = np.concatenate([f.mean(axis=2), std], axis=1).astype(np.float32)
        else:
            parts = [f[:, :, idx].mean(axis=2) for idx in np.array_split(np.arange(f.shape[2]), n)]
            out[n] = np.concatenate(parts + [std], axis=1).astype(np.float32)
    return out
