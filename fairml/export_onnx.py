"""Export the champion PyTorch model to ONNX (+ INT8) and assemble the deployable 'model bundle'.

Bundle (artifacts/model/): model.onnx, model_int8.onnx, preprocessor.joblib, metadata.json, constraints.txt

Exporter strategy (PyTorch changes this API often, so we try several and VALIDATE each result):
  1. torch.export based exporter  (dynamo=True, the modern default; needs `onnxscript`)
  2. legacy TorchScript exporter  (dynamo=False)
  3. very old torch (no `dynamo` argument)
Force one with ONNX_EXPORTER=dynamo|legacy.
"""
from __future__ import annotations

import collections
import datetime as dt
import importlib.metadata as md
import inspect
import json
import logging
import os
import platform
import shutil

import joblib
import numpy as np
import pandas as pd
import torch

from fairml import config as cfg
from fairml import drift
from fairml.gate import champion_variant
from fairml.model import ProbabilityHead, TabularMLP

logger = logging.getLogger(__name__)


def _load_torch(variant: str) -> TabularMLP:
    vdir = cfg.ARTIFACT_DIR / variant
    mc = json.loads((vdir / "model_config.json").read_text())
    model = TabularMLP(mc["in_features"], tuple(mc["hidden"]), mc["dropout"])
    model.load_state_dict(torch.load(vdir / "model.pt", map_location="cpu", weights_only=True))  # nosec B614
    return model.eval()


def _ort_run(path, X: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(None, {sess.get_inputs()[0].name: X.astype(np.float32)})[0].reshape(-1)


def _validate(path, model: TabularMLP, in_features: int) -> None:
    """The exported graph must match PyTorch AND accept any batch size (dynamic axis)."""
    rng = np.random.default_rng(0)
    for bs in (1, 3, 64):
        X = rng.normal(size=(bs, in_features)).astype(np.float32)
        with torch.no_grad():
            ref = torch.sigmoid(model(torch.from_numpy(X))).squeeze(-1).numpy()
        got = _ort_run(path, X)
        if got.shape != ref.shape or np.max(np.abs(got - ref)) > 1e-4:
            raise RuntimeError(f"exported model mismatch for batch size {bs}")


def _export(model: TabularMLP, in_features: int, path) -> str:
    head = ProbabilityHead(model).eval()
    params = inspect.signature(torch.onnx.export).parameters
    mode = os.getenv("ONNX_EXPORTER", "auto").lower()
    common = dict(input_names=["features"], output_names=["probability"])

    def dynamo_export():
        kw = dict(common, dynamo=True,
                  dynamic_shapes=({0: torch.export.Dim("batch")},))
        if "external_data" in params:
            kw["external_data"] = False
        torch.onnx.export(head, (torch.zeros(2, in_features),), str(path), **kw)

    def legacy_export():
        kw = dict(common, dynamic_axes={"features": {0: "batch"}, "probability": {0: "batch"}}, opset_version=17)
        if "dynamo" in params:
            kw["dynamo"] = False
        torch.onnx.export(head, (torch.zeros(1, in_features),), str(path), **kw)

    attempts = []
    if "dynamo" in params and mode in ("auto", "dynamo"):
        attempts.append(("dynamo", dynamo_export))
    if mode in ("auto", "legacy"):
        attempts.append(("legacy", legacy_export))

    errors = []
    for name, fn in attempts:
        try:
            if os.path.exists(path):
                os.remove(path)
            fn()
            _validate(path, model, in_features)
            logger.info("ONNX export succeeded with the '%s' exporter", name)
            return name
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            logger.warning("ONNX export with '%s' exporter failed: %s", name, exc)
    raise RuntimeError("All ONNX export strategies failed:\n  " + "\n  ".join(errors)
                       + "\nTip: `pip install -U onnx onnxscript` and re-run.")


def _quantize(src, dst) -> tuple[bool, dict]:
    try:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
        ops = collections.Counter(n.op_type for n in onnx.load(str(dst)).graph.node)
        n_q = ops.get("MatMulInteger", 0) + ops.get("QLinearMatMul", 0)
        if n_q == 0:
            logger.warning("INT8 model contains no quantised MatMul nodes (graph ops: %s) - no speed-up expected.",
                           dict(ops))
        return True, {"quantized_matmuls": n_q, "ops": dict(ops)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("INT8 quantisation skipped: %s", exc)
        return False, {}


def _env_snapshot() -> dict:
    pk = ["numpy", "pandas", "scikit-learn", "joblib", "scipy", "onnxruntime", "onnx", "torch", "fastapi", "pydantic"]
    versions = {}
    for p in pk:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def run(variant: str | None = None) -> dict:
    variant = variant or champion_variant()
    logger.info("Exporting champion variant '%s' to ONNX", variant)
    model = _load_torch(variant)
    in_features = json.loads((cfg.ARTIFACT_DIR / variant / "model_config.json").read_text())["in_features"]

    out = cfg.MODEL_DIR
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    fp32, int8 = out / "model.onnx", out / "model_int8.onnx"
    exporter = _export(model, in_features, fp32)
    has_int8, q_info = _quantize(fp32, int8)
    shutil.copy(cfg.ARTIFACT_DIR / "preprocessor.joblib", out / "preprocessor.joblib")
    pre = joblib.load(out / "preprocessor.joblib")

    # ---- parity checks on real test rows -------------------------------------------------
    test_df = pd.read_csv(cfg.ARTIFACT_DIR / "splits" / "test.csv").head(2000)
    X = pre.transform(test_df[cfg.FEATURES]).astype(np.float32)
    with torch.no_grad():
        p_torch = torch.sigmoid(model(torch.from_numpy(X))).squeeze(-1).numpy()
    p_onnx = _ort_run(fp32, X)
    parity = {"max_abs_diff_onnx_vs_torch": float(np.max(np.abs(p_torch - p_onnx)))}
    if has_int8:
        p_i8 = _ort_run(int8, X)
        parity["int8_label_agreement"] = float(np.mean((p_i8 >= 0.5) == (p_onnx >= 0.5)))
        parity["int8_max_abs_diff"] = float(np.max(np.abs(p_i8 - p_onnx)))
    logger.info("Parity checks: %s", parity)

    # ---- metadata (read by the API, the model card and /model-info) ----------------------
    train_df = pd.read_csv(cfg.ARTIFACT_DIR / "splits" / "train.csv")
    cat_enc = pre.named_transformers_["cat"]
    categories = {c: [str(v) for v in cats] for c, cats in zip(cfg.CATEGORICAL_FEATURES, cat_enc.categories_)}
    metrics = json.loads((cfg.ARTIFACT_DIR / variant / "metrics.json").read_text())
    fr = cfg.ARTIFACT_DIR / "fairness_report.json"
    champ = cfg.ARTIFACT_DIR / "champion.json"
    env = _env_snapshot()
    metadata = {
        "model_name": cfg.REGISTERED_MODEL_NAME, "model_version": cfg.MODEL_VERSION, "variant": variant,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "threshold": cfg.DECISION_THRESHOLD, "features": cfg.FEATURES,
        "numeric_features": cfg.NUMERIC_FEATURES, "categorical_features": cfg.CATEGORICAL_FEATURES,
        "categories": categories,
        "baselines": json.loads((cfg.ARTIFACT_DIR / "baselines.json").read_text()),
        "drift_reference": drift.build_reference(train_df, cfg.NUMERIC_FEATURES, cfg.CATEGORICAL_FEATURES),
        "has_int8": has_int8, "int8_info": q_info, "exporter": exporter,
        "protected_attribute": cfg.PROTECTED_ATTRIBUTE,
        "performance": metrics["performance"], "fairness": metrics["fairness"],
        "calibration": metrics.get("calibration"),
        "mitigation": metrics["params"].get("mitigation"),
        "trained_on": {k: metrics["device"].get(k) for k in ("device", "gpu_name", "torch", "cuda_build")},
        "champion": json.loads(champ.read_text()) if champ.exists() else None,
        "fairness_report": json.loads(fr.read_text()) if fr.exists() else None,
        "parity_checks": parity, "environment": env,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    # pip constraints so the Docker image installs the SAME numpy/pandas/scikit-learn that pickled the preprocessor
    pins = [f"{p}=={v}" for p, v in env["packages"].items() if p in ("numpy", "pandas", "scikit-learn", "joblib", "scipy")]
    (out / "constraints.txt").write_text("\n".join(pins) + "\n")
    logger.info("Model bundle ready in %s", out)
    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
