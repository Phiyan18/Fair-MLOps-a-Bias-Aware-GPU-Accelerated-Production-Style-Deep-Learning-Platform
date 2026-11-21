"""Lightweight inference engine used by the REST API (no torch / mlflow / shap needed)."""
from __future__ import annotations

import importlib.metadata as md
import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

from fairml import config as cfg

logger = logging.getLogger(__name__)


class Predictor:
    """Loads the model bundle (ONNX + preprocessor + metadata) and serves predictions."""

    def __init__(self, model_dir: str | Path | None = None, use_quantized: bool | None = None):
        self.model_dir = Path(model_dir) if model_dir else cfg.MODEL_DIR
        meta_path = self.model_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Model bundle not found in {self.model_dir}. Run `python -m fairml.pipeline` first."
            )
        self.metadata: dict[str, Any] = json.loads(meta_path.read_text())
        self.features: list[str] = self.metadata["features"]
        self.numeric: list[str] = self.metadata["numeric_features"]
        self.categorical: list[str] = self.metadata["categorical_features"]
        self.baselines: dict[str, Any] = self.metadata["baselines"]
        self.threshold: float = float(self.metadata["threshold"])
        self.version: str = self.metadata["model_version"]
        self._check_environment()
        self.preprocessor = joblib.load(self.model_dir / "preprocessor.joblib")

        if use_quantized is None:
            use_quantized = os.getenv("USE_QUANTIZED", "0").lower() in {"1", "true", "yes"}
        use_quantized = bool(use_quantized and self.metadata.get("has_int8"))
        self.model_file = "model_int8.onnx" if use_quantized else "model.onnx"

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "1"))
        self.session = ort.InferenceSession(
            str(self.model_dir / self.model_file), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def _check_environment(self) -> None:
        """The preprocessor is a pickled scikit-learn object: warn loudly if versions differ from training."""
        trained = self.metadata.get("environment", {}).get("packages", {})
        for pkg in ("scikit-learn", "numpy", "pandas"):
            try:
                now = md.version(pkg)
            except md.PackageNotFoundError:
                continue
            if pkg in trained and trained[pkg] != now:
                logger.warning("%s version differs from training (%s vs %s) - pickled preprocessor may misbehave. "
                               "Install with: pip install -c artifacts/model/constraints.txt ...", pkg, trained[pkg], now)

    # ------------------------------------------------------------------ helpers
    def _frame(self, records: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(records)[self.features].copy()
        for c in self.numeric:
            df[c] = df[c].astype(float)
        for c in self.categorical:
            df[c] = df[c].astype(str)
        return df

    def predict_proba_frame(self, df: pd.DataFrame) -> np.ndarray:
        X = self.preprocessor.transform(df[self.features]).astype(np.float32)
        return self.session.run(None, {self.input_name: X})[0].reshape(-1)

    def _label(self, p: float) -> str:
        return cfg.POSITIVE_LABEL if p >= self.threshold else cfg.NEGATIVE_LABEL

    # ------------------------------------------------------------------ public API
    def predict(self, records: list[dict]) -> list[dict]:
        probs = self.predict_proba_frame(self._frame(records))
        return [
            {"probability": float(p), "prediction": int(p >= self.threshold), "label": self._label(float(p))}
            for p in probs
        ]

    def explain(self, record: dict) -> dict:
        """Leave-one-feature-out attribution on the *raw* features.

        For each feature we replace its value by the 'typical person' baseline (median / mode
        of the training data) and measure how much the probability changes.
        contribution > 0  -> this feature pushes the prediction towards '>50K'.
        Cheap (one batched forward pass), model-agnostic and easy to explain to non-experts.
        For a rigorous global view see the offline SHAP analysis (fairml.explain).
        """
        base_row = {f: self.baselines[f] for f in self.features}
        rows = [dict(record)]
        for f in self.features:
            variant = dict(record)
            variant[f] = base_row[f]
            rows.append(variant)
        rows.append(base_row)  # the fully-typical person
        probs = self.predict_proba_frame(self._frame(rows))
        p = float(probs[0])
        contributions = [
            {"feature": f, "value": record[f], "contribution": float(p - probs[i + 1])}
            for i, f in enumerate(self.features)
        ]
        contributions.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        return {
            "probability": p,
            "prediction": int(p >= self.threshold),
            "label": self._label(p),
            "base_probability": float(probs[-1]),
            "contributions": contributions,
            "method": "leave-one-feature-out vs. typical-person baseline",
        }
