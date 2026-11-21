"""Shared fixtures.

IMPORTANT: environment variables are set BEFORE `fairml` is imported anywhere, so every test works in an
isolated temp directory and never touches your real ./artifacts or ./mlflow.db.
"""
import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="fairml_tests_"))
os.environ["ARTIFACT_DIR"] = str(_TMP / "artifacts")
os.environ["DATA_DIR"] = str(_TMP / "data")
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///" + (_TMP / "mlflow.db").as_posix()
os.environ["MLFLOW_ARTIFACT_ROOT"] = (_TMP / "mlartifacts").as_uri()
os.environ.pop("MODEL_DIR", None)
os.environ.pop("API_KEY", None)
os.environ.pop("API_KEYS", None)
os.environ.pop("RATE_LIMIT_PER_MIN", None)
os.environ["PRODUCTION_VARIANT"] = "auto"

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def artifacts():
    """Train tiny models on synthetic data and export the ONNX bundle (takes a few seconds)."""
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from fairml import calibration, export_onnx, fairness, gate, train

    train.run(synthetic=True, n_synthetic=3000, epochs=4, hidden=(32, 16), device="cpu", use_mlflow=False)
    fairness.run()
    calibration.run()
    gate.select_champion()
    export_onnx.run()
    return Path(os.environ["ARTIFACT_DIR"])
