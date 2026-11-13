"""Unit tests for the Stage 4 code (synthetic data only; never reported as results)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kwsba import run_stage4 as s4  # noqa: E402


def test_temperature_recovers_known_scale():
    rng = np.random.default_rng(0)
    n, c, T_true = 4000, 12, 3.0
    z = rng.standard_normal((n, c)) * 2.0
    p = np.exp(z / T_true); p /= p.sum(1, keepdims=True)
    y = np.array([rng.choice(c, p=pi) for pi in p])
    assert abs(s4.fit_temperature(z, y) - T_true) < 0.4


def test_conf_pred_is_a_probability_and_argmax():
    z = np.random.default_rng(1).standard_normal((50, 12))
    pred, conf = s4.conf_pred(z, 2.0)
    assert np.all(pred == z.argmax(1)) and np.all((conf > 0) & (conf <= 1))


def test_f1_from_cm_matches_sklearn():
    import analyze_stage4 as an
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(2)
    y = rng.integers(0, 12, 500); p = np.where(rng.random(500) < 0.6, y, rng.integers(0, 12, 500))
    assert abs(an.f1_from_cm(an.cm_w(y, p)) - f1_score(y, p, labels=list(range(12)), average="macro", zero_division=0)) < 1e-12


def test_bootstrap_detects_a_real_difference_and_not_a_null():
    import analyze_stage4 as an
    rng = np.random.default_rng(3)
    n = 600
    y = rng.integers(0, 12, n)
    good = np.where(rng.random(n) < 0.8, y, rng.integers(0, 12, n)).astype(np.int8)
    bad = np.where(rng.random(n) < 0.4, y, rng.integers(0, 12, n)).astype(np.int8)
    same = good.copy()
    data = {m: {"pred": {0: np.stack([p, p])}} for m, p in (("good", good), ("bad", bad), ("same", same))}
    o, lo, hi, p = an.boot_diff(y, [0], data, "good", "bad", [0, 1], 200, np.random.default_rng(0))
    assert o > 0.2 and lo > 0 and p < 0.02
    o, lo, hi, p = an.boot_diff(y, [0], data, "good", "same", [0, 1], 200, np.random.default_rng(0))
    assert o == 0 and lo == 0 and hi == 0


torch = pytest.importorskip("torch")


def test_tta_updates_only_bn_and_keeps_the_original_model():
    from kwsba.tcresnet import TCResNet8
    torch.manual_seed(0)
    m = TCResNet8(40, 12).eval()
    X = torch.randn(256, 40, 98) * 2 + 1
    before = {k: v.clone() for k, v in m.state_dict().items()}
    for meth in ("adabn", "tent", "adakws"):
        a = s4.tta_adapt(m, X, meth, 1e-2, 1, 0)
        for k, v in m.state_dict().items():                    # the source model is untouched
            assert torch.equal(v, before[k])
        mods = dict(m.named_modules())
        for (k, v), (_, w) in zip(m.named_parameters(), a.named_parameters()):
            if not isinstance(mods[k.rsplit(".", 1)[0]], torch.nn.BatchNorm1d):
                assert torch.equal(v, w), f"{meth}: non-BN parameter {k} changed"


def test_tent_lowers_entropy_on_the_adaptation_set():
    from kwsba.tcresnet import TCResNet8
    torch.manual_seed(1)
    m = TCResNet8(40, 12).eval()
    X = torch.randn(512, 40, 98)
    a0 = s4.tta_adapt(m, X, "adabn", 0, 0, 0)
    a1 = s4.tta_adapt(m, X, "tent", 1e-2, 3, 0)
    with torch.no_grad():
        e0 = s4.entropy(a0(X)).mean(); e1 = s4.entropy(a1(X)).mean()
    assert e1 < e0


def test_spec_mask_masks_within_limits():
    g = torch.Generator().manual_seed(0)
    x = torch.ones(8, 40, 98)
    y = s4.spec_mask(x, g)
    assert (y == 0).float().mean() > 0 and (y == 0).float().mean() < 0.4 and y.shape == x.shape


def test_reject_stats_reject_all_costs_lambda_and_full_coverage_equals_error_rate():
    from kwsba.run_reject_m import reject_stats
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=500)
    lg = rng.standard_normal((500, 3))
    flat = reject_stats(lg, y, T=1e6, lam=0.2)            # posteriors ~ 1/3 everywhere: everything is rejected
    assert flat["coverage"] == 0.0 and abs(flat["risk_chow"] - 0.2) < 1e-12
    sharp = reject_stats(np.eye(3)[rng.integers(0, 3, size=500)] * 50.0, y, T=1.0, lam=0.2)   # one-hot posteriors: everything is answered
    assert sharp["coverage"] == 1.0 and abs(sharp["risk_chow"] - sharp["risk_no_reject"]) < 1e-12
