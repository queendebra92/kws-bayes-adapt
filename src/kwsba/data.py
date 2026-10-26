"""Speech Commands v2 loading, official splits, and the 12-class benchmark.

Pre-registered decisions implemented here (fixed before any test result):
  * Splits are the official validation_list.txt / testing_list.txt; everything
    else is training. Speaker disjointness is asserted at cache-build time.
  * 12 classes: silence, unknown, and the ten keywords.
  * "unknown" keeps its natural frequency (all 25 non-target words); no
    re-balancing, so the ~17:1 imbalance discussed in the paper is preserved.
  * "silence" clips are 1-s cuts of the six _background_noise_ files, scaled
    by a random volume in [0, 1]. Each file is split in time (first 80 % ->
    train, next 10 % -> val, last 10 % -> test) so no sample is shared across
    splits. The number of silence clips per split equals the mean keyword
    class count of that split.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scipy.io import wavfile

try:  # progress bars are optional
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **_):
        return x

SR = 16000
L = 16000  # one second
KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABELS = ["silence", "unknown"] + KEYWORDS
SILENCE, UNKNOWN = 0, 1
SPLITS = ("train", "val", "test")
_SILENCE_RANGES = {"train": (0.0, 0.8), "val": (0.8, 0.9), "test": (0.9, 1.0)}


def label_of(word: str) -> int:
    return LABELS.index(word) if word in KEYWORDS else UNKNOWN


def read_wav_int16(path) -> np.ndarray:
    """Read a mono 16-kHz wav as int16, zero-padded / truncated to one second."""
    sr, x = wavfile.read(str(path))
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr}")
    if x.ndim > 1:
        x = x[:, 0]
    if x.dtype.kind == "f":
        x = np.clip(x, -1.0, 1.0) * 32767.0
    x = x.astype(np.int16)
    out = np.zeros(L, dtype=np.int16)
    n = min(len(x), L)
    out[:n] = x[:n]
    return out


def _read_lines(p: Path) -> set[str]:
    return {ln.strip().replace("\\", "/") for ln in p.read_text().splitlines() if ln.strip()}


def list_files(sc_root) -> dict[str, list[tuple[str, int]]]:
    """Return {split: [(relative_path, label), ...]} from the official lists."""
    root = Path(sc_root)
    val = _read_lines(root / "validation_list.txt")
    test = _read_lines(root / "testing_list.txt")
    out: dict[str, list[tuple[str, int]]] = {s: [] for s in SPLITS}
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for f in sorted(d.glob("*.wav")):
            rel = f"{d.name}/{f.name}"
            split = "val" if rel in val else "test" if rel in test else "train"
            out[split].append((rel, label_of(d.name)))
    return out


def speaker_of(rel: str) -> str:
    return Path(rel).name.split("_nohash_")[0]


def check_no_speaker_leakage(lists: dict[str, list[tuple[str, int]]]) -> None:
    spk = {s: {speaker_of(r) for r, _ in lists[s]} for s in SPLITS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = spk[a] & spk[b]
        if shared:
            raise AssertionError(f"speaker leakage between {a} and {b}: {sorted(shared)[:5]}")


def _silence_counts(lists) -> dict[str, int]:
    out = {}
    for s in SPLITS:
        n_kw = sum(1 for _, y in lists[s] if y >= 2)
        out[s] = int(round(n_kw / len(KEYWORDS)))
    return out


def make_silence(sc_root, counts: dict[str, int], data_seed: int) -> dict[str, np.ndarray]:
    files = sorted((Path(sc_root) / "_background_noise_").glob("*.wav"))
    if not files:
        raise FileNotFoundError("no _background_noise_/*.wav found")
    sigs = []
    for f in files:
        sr, x = wavfile.read(str(f))
        if x.ndim > 1:
            x = x[:, 0]
        if x.dtype.kind == "f":
            x = np.clip(x, -1.0, 1.0) * 32767.0
        sigs.append(x.astype(np.int16))
    rng = np.random.default_rng(data_seed)
    out = {}
    for split in SPLITS:
        a, b = _SILENCE_RANGES[split]
        segs = np.zeros((counts[split], L), dtype=np.int16)
        for i in range(counts[split]):
            sig = sigs[rng.integers(len(sigs))]
            lo, hi = int(a * len(sig)), int(b * len(sig))
            if hi - lo < L:
                raise ValueError("background-noise file too short for a held-out split")
            start = int(rng.integers(lo, hi - L + 1))
            vol = rng.uniform(0.0, 1.0)
            segs[i] = (sig[start:start + L].astype(np.float32) * vol).astype(np.int16)
        out[split] = segs
    return out


def build_cache(sc_root, cache_dir, data_seed: int = 0) -> dict:
    """Decode every wav once into {split}_wav.npy (int16, N x 16000) + {split}_y.npy."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    meta_path = cache / "meta.json"
    if meta_path.exists() and all((cache / f"{s}_wav.npy").exists() for s in SPLITS):
        return json.loads(meta_path.read_text())

    lists = list_files(sc_root)
    check_no_speaker_leakage(lists)
    counts = _silence_counts(lists)
    sil = make_silence(sc_root, counts, data_seed)
    root = Path(sc_root)
    meta = {"data_seed": data_seed, "splits": {}}
    for split in SPLITS:
        items = lists[split]
        n = len(items) + counts[split]
        wav = open_memmap(cache / f"{split}_wav.npy", mode="w+", dtype=np.int16, shape=(n, L))
        y = np.zeros(n, dtype=np.int8)
        for i, (rel, lab) in enumerate(tqdm(items, desc=f"decode {split}")):
            wav[i] = read_wav_int16(root / rel)
            y[i] = lab
        wav[len(items):] = sil[split]
        y[len(items):] = SILENCE
        wav.flush()
        del wav
        np.save(cache / f"{split}_y.npy", y)
        meta["splits"][split] = {
            "n": int(n),
            "per_class": {LABELS[c]: int((y == c).sum()) for c in range(len(LABELS))},
        }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def load_split(cache_dir, split: str):
    """Return (wav memmap int16 [N, L], labels int [N])."""
    cache = Path(cache_dir)
    wav = np.load(cache / f"{split}_wav.npy", mmap_mode="r")
    y = np.load(cache / f"{split}_y.npy").astype(np.int64)
    return wav, y
