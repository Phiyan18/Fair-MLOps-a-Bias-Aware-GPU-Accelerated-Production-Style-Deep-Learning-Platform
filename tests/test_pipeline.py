import json

import numpy as np


def test_artifacts_created(artifacts):
    for rel in ("baseline/model.pt", "reweighed/model.pt", "adversarial/model.pt", "fairness_report.json",
                "champion.json", "calibration_report.json", "model/model.onnx", "model/preprocessor.joblib",
                "model/metadata.json", "model/constraints.txt"):
        assert (artifacts / rel).exists(), rel


def test_fairness_report_covers_all_variants(artifacts):
    rep = json.loads((artifacts / "fairness_report.json").read_text())
    assert set(rep["variants"]) == {"baseline", "reweighed", "adversarial"}
    assert set(rep["improvement_pct"]) == {"reweighed", "adversarial"}


def test_onnx_matches_torch_and_metadata(artifacts):
    meta = json.loads((artifacts / "model" / "metadata.json").read_text())
    assert meta["parity_checks"]["max_abs_diff_onnx_vs_torch"] < 1e-4
    assert meta["variant"] in {"baseline", "reweighed", "adversarial"}
    assert "drift_reference" in meta and "environment" in meta


def test_predictor_probabilities_valid(artifacts):
    from fairml.inference import Predictor

    p = Predictor(use_quantized=False)
    rec = {"age": 40, "education_num": 13, "capital_gain": 0.0, "capital_loss": 0.0, "hours_per_week": 45,
           "workclass": "Private", "marital_status": "Married-civ-spouse", "occupation": "Exec-managerial",
           "relationship": "Husband", "race": "White", "sex": "Male", "native_country": "United-States"}
    out = p.predict([rec, dict(rec, sex="Female")])
    assert len(out) == 2 and all(0.0 <= o["probability"] <= 1.0 for o in out)
    ex = p.explain(rec)
    assert len(ex["contributions"]) == len(p.features)
    assert np.isfinite([c["contribution"] for c in ex["contributions"]]).all()


def test_gate_report_runs(artifacts):
    from fairml import gate

    rep = gate.release_gate(fail=False)
    assert "checks" in rep and isinstance(rep["passed"], bool)
