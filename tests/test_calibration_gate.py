import json

import numpy as np
import pytest

from fairml import calibration


def test_ece_perfect_and_bad():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)            # perfectly calibrated by construction
    assert calibration.ece(y, p) < 0.02
    assert calibration.ece(np.zeros(1000), np.full(1000, 0.9)) == pytest.approx(0.9)


def test_brier():
    assert calibration.brier([1, 0], [1.0, 0.0]) == 0.0
    assert calibration.brier([1, 0], [0.0, 1.0]) == 1.0


def _write_variant(root, name, acc, auc, di, spd, eod):
    d = root / name
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({
        "performance": {"accuracy": acc, "roc_auc": auc},
        "fairness": {"disparate_impact": di, "statistical_parity_difference": spd, "equal_opportunity_difference": eod},
        "calibration": {"ece": 0.03}, "params": {"synthetic": False}}))


def test_champion_prefers_fair_variant(tmp_path, monkeypatch):
    from fairml import config as cfg
    from fairml import gate

    monkeypatch.setattr(cfg, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(cfg, "PRODUCTION_VARIANT", "auto")
    _write_variant(tmp_path, "baseline", 0.85, 0.91, 0.30, -0.19, -0.10)
    _write_variant(tmp_path, "reweighed", 0.84, 0.90, 0.90, -0.03, -0.04)
    _write_variant(tmp_path, "adversarial", 0.83, 0.89, 0.95, -0.02, -0.03)
    out = gate.select_champion()
    assert out["champion"] == "reweighed"          # fair + highest AUC among fair variants
    assert out["candidates"]["baseline"]["meets_fairness_criteria"] is False


def test_champion_fallback_when_none_fair(tmp_path, monkeypatch):
    from fairml import config as cfg
    from fairml import gate

    monkeypatch.setattr(cfg, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(cfg, "PRODUCTION_VARIANT", "auto")
    _write_variant(tmp_path, "baseline", 0.85, 0.91, 0.30, -0.19, -0.20)
    _write_variant(tmp_path, "reweighed", 0.84, 0.90, 0.60, -0.08, -0.15)
    _write_variant(tmp_path, "adversarial", 0.70, 0.80, 0.70, -0.01, -0.01)   # too much accuracy lost
    out = gate.select_champion()
    assert out["champion"] == "reweighed"


def test_forced_champion(tmp_path, monkeypatch):
    from fairml import config as cfg
    from fairml import gate

    monkeypatch.setattr(cfg, "ARTIFACT_DIR", tmp_path)
    _write_variant(tmp_path, "baseline", 0.85, 0.91, 0.30, -0.19, -0.10)
    assert gate.select_champion("baseline")["champion"] == "baseline"
    with pytest.raises(ValueError):
        gate.select_champion("does-not-exist")
