"""Central configuration. Everything can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", ROOT / "artifacts"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ARTIFACT_DIR / "model"))

# MLflow >= 3.7 uses SQLite by default; the old file store (./mlruns) is in maintenance mode.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///" + (ROOT / "mlflow.db").as_posix())
MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", (ROOT / "mlartifacts").as_uri())

EXPERIMENT_NAME = "adult-income-fair-mlp"
REGISTERED_MODEL_NAME = "adult-income-classifier"
MODEL_VERSION = "2.0.0"
SEED = int(os.getenv("SEED", "42"))

# ---------------------------------------------------------------- dataset schema
NUMERIC_FEATURES = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_FEATURES = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "income"  # 1 if income > 50K else 0
POSITIVE_LABEL = ">50K"
NEGATIVE_LABEL = "<=50K"

# ---------------------------------------------------------------- fairness setup
PROTECTED_ATTRIBUTE = "sex"
PRIVILEGED_VALUE = "Male"
UNPRIVILEGED_VALUE = "Female"

DECISION_THRESHOLD = 0.5

# baseline        : plain training
# reweighed       : pre-processing mitigation (AIF360 Reweighing)
# adversarial     : in-processing mitigation (adversary + gradient reversal, PyTorch)
VARIANTS = ("baseline", "reweighed", "adversarial")
# "auto" = the pipeline picks the champion (see fairml/gate.py); or force one variant name.
PRODUCTION_VARIANT = os.getenv("PRODUCTION_VARIANT", "auto")

# ---------------------------------------------------------------- release-gate thresholds
def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


GATE_MIN_ACCURACY = _f("GATE_MIN_ACCURACY", 0.80)
GATE_MIN_AUC = _f("GATE_MIN_AUC", 0.85)
GATE_MIN_DI = _f("GATE_MIN_DI", 0.80)          # "80% rule"
GATE_MAX_DI = _f("GATE_MAX_DI", 1.25)
GATE_MAX_ABS_EOD = _f("GATE_MAX_ABS_EOD", 0.10)  # |equal opportunity difference|
GATE_MAX_ECE = _f("GATE_MAX_ECE", 0.10)          # calibration error
GATE_MIN_INT8_AGREEMENT = _f("GATE_MIN_INT8_AGREEMENT", 0.98)
GATE_MAX_ACC_DROP = _f("GATE_MAX_ACC_DROP", 0.05)  # fairness may cost at most 5 accuracy points
