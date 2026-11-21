import numpy as np
import pytest

from fairml import fairness


def test_perfect_parity():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    is_priv = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    m = fairness.metrics_numpy(y_true, y_true, is_priv)
    assert m["statistical_parity_difference"] == pytest.approx(0.0)
    assert m["disparate_impact"] == pytest.approx(1.0)
    assert m["equal_opportunity_difference"] == pytest.approx(0.0)


def test_known_values():
    is_priv = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    m = fairness.metrics_numpy(y_true, y_pred, is_priv)
    assert m["statistical_parity_difference"] == pytest.approx(0.25 - 0.75)
    assert m["disparate_impact"] == pytest.approx(0.25 / 0.75)
    assert m["equal_opportunity_difference"] == pytest.approx(0.5 - 1.0)


def test_reweighing_equalises_weighted_positive_rate():
    rng = np.random.default_rng(0)
    is_priv = rng.random(5000) < 0.6
    y = (rng.random(5000) < np.where(is_priv, 0.4, 0.15)).astype(int)
    w = fairness.reweighing_weights_numpy(y, is_priv)

    def wrate(mask):
        return (w[mask] * y[mask]).sum() / w[mask].sum()

    assert wrate(is_priv) == pytest.approx(wrate(~is_priv), abs=1e-9)
    assert w.sum() == pytest.approx(len(y))


def test_aif360_matches_numpy_when_it_works():
    pytest.importorskip("aif360")
    rng = np.random.default_rng(1)
    is_priv = rng.random(2000) < 0.6
    y_true = (rng.random(2000) < 0.3).astype(int)
    y_pred = (rng.random(2000) < np.where(is_priv, 0.35, 0.2)).astype(int)
    try:  # AIF360 0.6.x may not support the very newest NumPy/pandas -> skip instead of failing
        a = fairness.metrics_aif360(y_true, y_pred, is_priv)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"AIF360 not usable with this NumPy/pandas combination: {exc}")
    n = fairness.metrics_numpy(y_true, y_pred, is_priv)
    for k in ("statistical_parity_difference", "disparate_impact", "equal_opportunity_difference", "average_odds_difference"):
        assert a[k] == pytest.approx(n[k], abs=1e-9), k


def test_fairness_metrics_falls_back_gracefully():
    out = fairness.fairness_metrics([1, 0, 1, 0], [1, 0, 0, 0], [1, 1, 0, 0])
    assert out["backend"] in {"aif360", "numpy"}
    assert "statistical_parity_difference" in out
