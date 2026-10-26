import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kwsba import data, features, noise  # noqa: E402
from kwsba.bayes import REJECT, GaussianBayes  # noqa: E402


@pytest.fixture(scope="session")
def fake(tmp_path_factory):
    import make_fake_data
    root = tmp_path_factory.mktemp("fake")
    make_fake_data.main(root)
    return root


# ---------------------------------------------------------------- features
def test_mfcc_shapes():
    x = np.random.default_rng(0).standard_normal((3, data.L)).astype(np.float32)
    assert features.mfcc(x, 13).shape == (3, 13, 98)
    assert features.mfcc(x, 40).shape == (3, 40, 98)
    assert features.pooled_features(x).shape == (3, 78)


@pytest.mark.skipif(importlib.util.find_spec("librosa") is None, reason="librosa not installed")
def test_mel_filterbank_matches_librosa():
    import librosa
    fb = librosa.filters.mel(sr=data.SR, n_fft=features.NFFT, n_mels=features.N_MELS,
                             fmin=features.FMIN, fmax=features.FMAX, htk=True, norm=None)
    assert np.allclose(features.mel_filterbank(), fb, atol=1e-5)


def test_log_mel_matches_manual_reference():
    """Independent frame-by-frame computation of the first frame."""
    x = np.random.default_rng(1).standard_normal(data.L).astype(np.float32) * 0.1
    ours = features.log_mel(x[None])[0]
    pre = np.append(x[0], x[1:] - features.PREEMPH * x[:-1])
    frame = pre[:features.WIN] * np.hamming(features.WIN)
    spec = np.abs(np.fft.rfft(frame, n=features.NFFT)) ** 2
    ref = np.log(features.mel_filterbank() @ spec + 1e-10)
    assert np.allclose(ours[0], ref, atol=1e-3)


def test_deltas_of_ramp_is_constant():
    ramp = np.arange(20, dtype=np.float32)[None, None, :]
    d = features.deltas(ramp)
    assert np.allclose(d[0, 0, 3:-3], 1.0, atol=1e-5)


# ------------------------------------------------------------------- noise
@pytest.mark.parametrize("snr", [20, 10, 0])
def test_mix_hits_target_snr(snr):
    rng = np.random.default_rng(0)
    clean = (0.1 * rng.standard_normal((8, data.L))).astype(np.float32)
    nz = rng.standard_normal((8, data.L)).astype(np.float32)
    mixed = noise.mix_at_snr(clean, nz, snr)
    added = (mixed - clean).astype(np.float64)
    meas = 10 * np.log10(np.mean(clean.astype(np.float64) ** 2, 1) / np.mean(added ** 2, 1))
    assert np.allclose(meas, snr, atol=0.05)


def test_noise_pool_partitions_are_disjoint(fake):
    pool = noise.NoisePool(fake / "musan")
    for kind in ("environmental", "speech"):
        parts = pool.files[kind]
        sets = [set(parts[p]) for p in ("train", "val", "test")]
        assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
        assert sum(len(s) for s in sets) == 20


def test_all_noise_kinds_sample(fake):
    pool = noise.NoisePool(fake / "musan")
    rng = np.random.default_rng(0)
    for kind in noise.NOISE_KINDS:
        x = pool.sample(kind, 4, rng, "test")
        assert x.shape == (4, data.L) and np.isfinite(x).all() and x.std() > 0


def test_noise_is_reproducible_per_seed(fake):
    pool = noise.NoisePool(fake / "musan")
    a = pool.sample("babble", 3, np.random.default_rng([1, 0, 10]), "test")
    b = pool.sample("babble", 3, np.random.default_rng([1, 0, 10]), "test")
    c = pool.sample("babble", 3, np.random.default_rng([2, 0, 10]), "test")
    assert np.array_equal(a, b) and not np.array_equal(a, c)


# -------------------------------------------------------------------- data
def test_cache_split_integrity(fake, tmp_path):
    meta = data.build_cache(fake / "speech_commands", tmp_path / "cache")
    for s in data.SPLITS:
        wav, y = data.load_split(tmp_path / "cache", s)
        assert wav.shape == (meta["splits"][s]["n"], data.L) and len(y) == len(wav)
        assert set(np.unique(y)) <= set(range(12))
    n_tr = meta["splits"]["train"]["per_class"]
    assert n_tr["unknown"] > n_tr["yes"]          # natural imbalance preserved
    assert n_tr["silence"] == round(sum(n_tr[k] for k in data.KEYWORDS) / 10)


def test_speaker_leakage_is_detected():
    lists = {"train": [("a/spk1_nohash_0.wav", 2)], "val": [("a/spk2_nohash_0.wav", 2)],
             "test": [("b/spk1_nohash_3.wav", 2)]}
    with pytest.raises(AssertionError):
        data.check_no_speaker_leakage(lists)


# ------------------------------------------------------------------- bayes
def test_gaussian_bayes_matches_analytic_bayes_error():
    from scipy.stats import norm
    rng = np.random.default_rng(0)
    n = 40000
    y = rng.integers(2, size=n)
    X = rng.standard_normal((n, 1)) + 2.0 * y[:, None]       # means 0 and 2, unit variance
    m = GaussianBayes(eps=1e-6).fit(X[: n // 2], y[: n // 2])
    acc = np.mean(m.predict(X[n // 2:]) == y[n // 2:])
    assert abs(acc - norm.cdf(1.0)) < 0.01                    # optimal accuracy = Phi(d/2)


def test_chow_rejection_monotone_and_correct():
    rng = np.random.default_rng(1)
    y = rng.integers(2, size=6000)
    X = rng.standard_normal((6000, 2)) + 1.5 * y[:, None]
    m = GaussianBayes(eps=1e-6).fit(X, y)
    rej = [np.mean(m.predict_reject(X, lam) == REJECT) for lam in (0.5, 0.3, 0.1, 0.02)]
    assert rej == sorted(rej)                                 # smaller lambda_r rejects more
    assert rej[0] == 0.0                                      # lambda_r = 0.5 never rejects 2 classes
    p = m.predict_reject(X, 0.2)
    kept = p != REJECT
    assert np.mean(p[kept] == y[kept]) > np.mean(m.predict(X) == y)   # accuracy rises on kept samples


# -------------------------------------------------------- end-to-end smoke
def test_run_baselines_smoke(fake, tmp_path):
    from kwsba import run_baselines
    run_baselines.main(["--sc-root", str(fake / "speech_commands"),
                        "--musan-root", str(fake / "musan"),
                        "--cache-dir", str(tmp_path / "c"), "--out", str(tmp_path / "r"),
                        "--n-seeds", "2"])
    for f in ("per_run.csv", "table2_by_snr.csv", "table2_macro_f1.tex", "run_info.json"):
        assert (tmp_path / "r" / f).exists()
    import pandas as pd
    df = pd.read_csv(tmp_path / "r" / "per_run.csv")
    maj = df[df.method == "B3 majority class"]
    assert maj.macro_f1.max() < 0.2                           # floor really is a floor
    clean = df[(df.condition == "clean") & df.method.str.startswith("B1 ")].macro_f1.iloc[0]
    assert clean > 0.5                                        # pipeline learns the synthetic classes


def test_augmentation_noise_differs_between_batches(fake):
    """Regression: the generator must be shared across batches, not re-created per batch."""
    from kwsba import run_baselines
    pool = noise.NoisePool(fake / "musan")
    tf = run_baselines.aug_transform(pool, 777, "train")
    clean = np.zeros((16, data.L), dtype=np.int16) + 500
    a, b = tf(clean), tf(clean)
    assert not np.allclose(a, b)
    tf2 = run_baselines.aug_transform(pool, 777, "train")
    assert np.allclose(tf2(clean), a)          # but reproducible from the same seed


# ---------------------------------------------------------------- TC-ResNet8
torch = pytest.importorskip("torch")


def test_tcresnet8_shape_and_size():
    from kwsba.tcresnet import TCResNet8, n_params
    m = TCResNet8(40, 12)
    y = m(torch.randn(5, 40, 98))
    assert y.shape == (5, 12)
    assert 55_000 < n_params(m) < 80_000          # published TC-ResNet8 is ~66k parameters


def test_time_shift_zero_fills_and_preserves_shape():
    from kwsba.tcresnet import time_shift
    x = torch.ones(64, 3, 20)
    y = time_shift(x, 4)
    assert y.shape == x.shape
    assert set(y.unique().tolist()) <= {0.0, 1.0}
    assert (y == 0).any() and (y == 1).any()      # some samples shifted, none fully lost
    assert torch.equal(time_shift(x, 0), x)


def test_tcresnet_can_overfit_a_tiny_batch():
    from kwsba.tcresnet import TCResNet8
    torch.manual_seed(0)
    m = TCResNet8(40, 12)
    x, y = torch.randn(48, 40, 98), torch.randint(12, (48,))
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for _ in range(80):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss)
    assert float(loss) < 0.3 * first


@pytest.mark.parametrize("train_set", ["clean", "aug"])
def test_run_tcresnet_smoke(fake, tmp_path, train_set):
    from kwsba import run_baselines, run_tcresnet
    run_baselines.main(["--sc-root", str(fake / "speech_commands"), "--musan-root", str(fake / "musan"),
                        "--cache-dir", str(tmp_path / "c"), "--out", str(tmp_path / "r"), "--n-seeds", "1",
                        "--pca-dim", "40"])
    run_tcresnet.main(["--musan-root", str(fake / "musan"), "--cache-dir", str(tmp_path / "c"),
                       "--out", str(tmp_path / "t"), "--n-seeds", "2", "--epochs", "3",
                       "--batch", "64", "--train-set", train_set])
    import pandas as pd
    df = pd.read_csv(tmp_path / "t" / f"per_run_{train_set}.csv")
    assert set(df.seed) == {0, 1} and df.condition.nunique() == 10
    assert (tmp_path / "t" / f"run_info_{train_set}.json").exists()
    ref = pd.read_csv(tmp_path / "r" / "per_run.csv")
    assert set(df.columns) == set(ref.columns)    # same schema as the Bayes baselines


# ----------------------------------------------------------- EM adaptation
def _shifted_problem(seed=0, n=4000, shift=(1.2, 1.2)):
    rng = np.random.default_rng(seed)
    def draw():
        y = rng.integers(2, size=n)
        return rng.standard_normal((n, 2)) + np.array([3.0, 0.0]) * y[:, None], y
    (X, y), (X2, y2) = draw(), draw()
    return X, y, X2 + np.array(shift), y2


def test_em_adaptation_recovers_accuracy_under_shift():
    from kwsba.bayes import adapt_em
    X, y, X2, y2 = _shifted_problem()
    m = GaussianBayes(1e-6).fit(X, y)
    fixed = np.mean(m.predict(X2[2000:]) == y2[2000:])
    adapted = np.mean(adapt_em(m, X2[:2000], tau=16, n_iter=10).predict(X2[2000:]) == y2[2000:])
    assert adapted > fixed + 0.08


def test_em_adaptation_limits_and_no_labels_needed():
    from kwsba.bayes import adapt_em
    X, y, X2, y2 = _shifted_problem(1)
    m = GaussianBayes(1e-6).fit(X, y)
    assert np.allclose(adapt_em(m, X2, tau=1e9, n_iter=5).mu_, m.mu_, atol=1e-3)      # huge prior -> unchanged
    assert np.allclose(adapt_em(m, X2, tau=16, n_iter=0).mu_, m.mu_)                  # zero iterations -> unchanged
    a = adapt_em(m, X2, tau=16, n_iter=5)
    assert np.all(np.linalg.eigvalsh(a.cov_[0]) > 0)                                  # covariances stay PD
    assert m.mu_[0, 0] != a.mu_[0, 0]                                                 # original untouched copy semantics


def test_em_adaptation_does_not_harm_matched_data():
    from kwsba.bayes import adapt_em
    X, y, _, _ = _shifted_problem(2)
    m = GaussianBayes(1e-6).fit(X[:2000], y[:2000])
    before = np.mean(m.predict(X[2000:]) == y[2000:])
    after = np.mean(adapt_em(m, X[2000:3000], tau=16, n_iter=10).predict(X[3000:]) == y[3000:])
    assert after > before - 0.02


def test_reject_stats_consistent():
    from kwsba.run_adapt import reject_stats
    X, y, _, _ = _shifted_problem(3)
    m = GaussianBayes(1e-6).fit(X, y)
    r = reject_stats(m, X, y, 0.2)
    assert 0 < r["coverage"] < 1 and r["acc_accepted"] > np.mean(m.predict(X) == y)
    r0 = reject_stats(m, X, y, 0.5)
    assert r0["coverage"] == 1.0 and abs(r0["risk"] - (1 - np.mean(m.predict(X) == y))) < 1e-9


def test_run_adapt_smoke(fake, tmp_path):
    from kwsba import run_baselines, run_adapt
    run_baselines.main(["--sc-root", str(fake / "speech_commands"), "--musan-root", str(fake / "musan"),
                        "--cache-dir", str(tmp_path / "c"), "--out", str(tmp_path / "r"), "--n-seeds", "1",
                        "--pca-dim", "40", "--eps", "0.01"])
    run_adapt.main(["--musan-root", str(fake / "musan"), "--cache-dir", str(tmp_path / "c"),
                    "--out", str(tmp_path / "a"), "--n-seeds", "2", "--n-adapt", "20", "40",
                    "--pca-dim", "40", "--tune-tau", "4", "64", "--tune-tau2", "256", "1024",
                    "--tune-n", "30"])
    import pandas as pd
    d = pd.read_csv(tmp_path / "a" / "per_run_adapt.csv")
    assert d.method.nunique() == 1 + 3 * 2 and set(d.seed) == {0, 1}
    # the 'fixed' rows must reproduce B1 from run_baselines exactly (same noisy test audio)
    b1 = pd.read_csv(tmp_path / "r" / "per_run.csv")
    b1 = b1[b1.method == "B1 Gaussian Bayes (clean-trained)"]
    f = d[d.method == "B1 Gaussian Bayes (clean-trained)"]
    mm = f[(f.seed == 0)].merge(b1[b1.seed == 0], on=["condition"], suffixes=("_a", "_b"))
    assert len(mm) == 10 and np.allclose(mm.macro_f1_a, mm.macro_f1_b)
    assert (tmp_path / "a" / "reject_curve.csv").exists()


def test_shared_affine_adaptation_beats_per_class_em_under_feature_shift():
    from kwsba.bayes import adapt_em, fit_affine
    rng = np.random.default_rng(1)
    d = 20
    def draw(n, shift=False):
        y = rng.integers(3, size=n)
        X = rng.standard_normal((n, d)) + np.eye(d)[y] * 2.0
        if shift:
            X = X * 0.5 + np.where(np.arange(d) % 2 == 0, 2.0, -1.5)
        return X, y
    X, y = draw(8000)
    m = GaussianBayes(0.01).fit(X, y, 3)
    Xs, ys = draw(4000, True)
    acc = lambda mod: np.mean(mod.predict(Xs[2000:]) == ys[2000:])        # noqa: E731
    fixed = acc(m)
    aff = acc(fit_affine(m, Xs[:500]))
    assert aff > fixed + 0.15
    Xm, ym = draw(4000)                                                   # matched data: no harm
    assert abs(np.mean(fit_affine(m, draw(1000)[0]).predict(Xm) == ym) - np.mean(m.predict(Xm) == ym)) < 0.02
