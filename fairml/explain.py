"""Global explainability with SHAP (offline analysis, not part of the serving image).

We explain the model at the level of the 12 RAW input features (not the ~90 one-hot columns),
which is what a stakeholder actually cares about. Categorical values are integer-coded so
SHAP's KernelExplainer can swap them with background values.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from fairml import config as cfg
from fairml.inference import Predictor

logger = logging.getLogger(__name__)


def run(n_background: int = 25, n_explain: int = 80, nsamples: int = 300) -> dict | None:
    try:
        import shap
    except ImportError:
        logger.warning("shap is not installed - skipping SHAP analysis.")
        return None

    predictor = Predictor(use_quantized=False)
    cats = predictor.metadata["categories"]
    features = predictor.features
    test_df = pd.read_csv(cfg.ARTIFACT_DIR / "splits" / "test.csv")
    rng = np.random.default_rng(cfg.SEED)

    def encode(df: pd.DataFrame) -> np.ndarray:
        cols = []
        for f in features:
            if f in cats:
                lookup = {v: i for i, v in enumerate(cats[f])}
                cols.append(df[f].map(lambda v: lookup.get(str(v), 0)).astype(float).values)
            else:
                cols.append(df[f].astype(float).values)
        return np.column_stack(cols)

    def decode(M: np.ndarray) -> pd.DataFrame:
        data = {}
        for j, f in enumerate(features):
            if f in cats:
                idx = np.clip(np.rint(M[:, j]).astype(int), 0, len(cats[f]) - 1)
                data[f] = np.array(cats[f], dtype=object)[idx]
            else:
                data[f] = M[:, j].astype(float)
        return pd.DataFrame(data)

    def f(M: np.ndarray) -> np.ndarray:
        return predictor.predict_proba_frame(decode(np.atleast_2d(M)))

    background = encode(test_df.sample(n_background, random_state=cfg.SEED))
    X = encode(test_df.sample(n_explain, random_state=cfg.SEED + 1))
    explainer = shap.KernelExplainer(f, background)
    sv = explainer.shap_values(X, nsamples=nsamples, silent=True)
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 0]

    importance = np.abs(sv).mean(axis=0)
    order = np.argsort(importance)[::-1]
    result = {
        "method": "SHAP KernelExplainer on raw features",
        "n_explained": int(n_explain),
        "mean_abs_shap": {features[i]: float(importance[i]) for i in order},
    }
    sex_idx = features.index(cfg.PROTECTED_ATTRIBUTE)
    result["protected_attribute_rank"] = int(np.where(order == sex_idx)[0][0]) + 1
    (cfg.ARTIFACT_DIR / "shap_importance.json").write_text(json.dumps(result, indent=2))
    _plot(features, importance, order)
    logger.info("Top drivers: %s", ", ".join(features[i] for i in order[:5]))
    return result


def _plot(features, importance, order) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [features[i] for i in order][::-1]
    vals = [importance[i] for i in order][::-1]
    colors = ["#d62728" if n == cfg.PROTECTED_ATTRIBUTE else "#1f77b4" for n in names]
    ax.barh(names, vals, color=colors)
    ax.set_xlabel("mean |SHAP value|  (impact on P(income > 50K))")
    ax.set_title("Global feature importance (protected attribute in red)")
    fig.tight_layout()
    fig.savefig(cfg.ARTIFACT_DIR / "shap_importance.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
