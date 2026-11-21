import numpy as np
import pandas as pd

from fairml import config as cfg
from fairml import data, drift


def _records(df):
    return df[cfg.FEATURES].to_dict("records")


def test_no_drift_on_same_distribution():
    train = data.make_synthetic(4000, seed=1)
    live = data.make_synthetic(1000, seed=2)
    ref = drift.build_reference(train, cfg.NUMERIC_FEATURES, cfg.CATEGORICAL_FEATURES)
    rep = drift.drift_report(ref, _records(live))
    assert rep["status"] == "ok"
    assert rep["max_psi"] < drift.WARN_PSI


def test_detects_shifted_numeric_and_new_category():
    train = data.make_synthetic(4000, seed=1)
    ref = drift.build_reference(train, cfg.NUMERIC_FEATURES, cfg.CATEGORICAL_FEATURES)
    live = data.make_synthetic(500, seed=3).copy()
    live["age"] = live["age"] + 25
    live["occupation"] = "Brand-New-Occupation"
    rep = drift.drift_report(ref, _records(live))
    assert rep["status"] == "alert"
    assert rep["features"]["age"]["status"] == "alert"
    assert rep["features"]["occupation"]["status"] == "alert"


def test_insufficient_data():
    train = data.make_synthetic(1000)
    ref = drift.build_reference(train, cfg.NUMERIC_FEATURES, cfg.CATEGORICAL_FEATURES)
    assert drift.drift_report(ref, _records(train.head(5)))["status"] == "insufficient_data"


def test_psi_identical_is_zero_and_spike_at_zero_handled():
    p = np.array([0.2, 0.3, 0.5])
    assert drift.psi(p, p) == 0.0
    df = pd.DataFrame({"x": [0.0] * 90 + list(range(1, 11))})
    ref = drift.build_reference(df, ["x"], [])
    # capital_gain-like column: share of zeros must be captured in its own bin
    assert max(ref["numeric"]["x"]["proportions"]) >= 0.89


def test_monitor_window():
    m = drift.DriftMonitor(window=10)
    m.add([{"a": i} for i in range(25)])
    assert len(m.snapshot()) == 10
