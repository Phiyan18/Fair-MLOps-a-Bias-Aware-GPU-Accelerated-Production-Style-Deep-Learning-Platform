"""Fairness metrics + bias mitigation (reweighing).

Primary backend: IBM's open-source AIF360 toolkit.
Fallback backend: a small NumPy implementation of the *same* definitions, so
the project still works if AIF360 cannot be installed. Tests check that the
two backends agree.

Definitions (unprivileged group = Female, privileged = Male, favourable = income > 50K):
  statistical_parity_difference = P(y_hat=1 | unpriv) - P(y_hat=1 | priv)        ideal 0
  disparate_impact              = P(y_hat=1 | unpriv) / P(y_hat=1 | priv)        ideal 1 (0.8-1.25 = "80% rule")
  equal_opportunity_difference  = TPR(unpriv) - TPR(priv)                        ideal 0
  average_odds_difference       = 0.5 * [(FPR_u - FPR_p) + (TPR_u - TPR_p)]      ideal 0
"""
from __future__ import annotations

import json
import logging

import numpy as np

from fairml import config as cfg

logger = logging.getLogger(__name__)

_AIF360_BROKEN = False  # set after the first failure so we only warn once

SA = cfg.PROTECTED_ATTRIBUTE
_UNPRIV = [{SA: 0}]
_PRIV = [{SA: 1}]


def _rate(mask_num: np.ndarray, mask_den: np.ndarray) -> float:
    den = mask_den.sum()
    return float((mask_num & mask_den).sum() / den) if den > 0 else float("nan")


# ------------------------------------------------------------------ metrics
def metrics_numpy(y_true, y_pred, is_priv) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    priv = np.asarray(is_priv).astype(bool)
    unp = ~priv

    sel_u = _rate(y_pred == 1, unp)
    sel_p = _rate(y_pred == 1, priv)
    tpr_u = _rate(y_pred == 1, unp & (y_true == 1))
    tpr_p = _rate(y_pred == 1, priv & (y_true == 1))
    fpr_u = _rate(y_pred == 1, unp & (y_true == 0))
    fpr_p = _rate(y_pred == 1, priv & (y_true == 0))

    return {
        "statistical_parity_difference": sel_u - sel_p,
        "disparate_impact": (sel_u / sel_p) if sel_p and sel_p > 0 else float("nan"),
        "equal_opportunity_difference": tpr_u - tpr_p,
        "average_odds_difference": 0.5 * ((fpr_u - fpr_p) + (tpr_u - tpr_p)),
        "selection_rate_unprivileged": sel_u,
        "selection_rate_privileged": sel_p,
    }


def _aif360_dataset(y, is_priv):
    import pandas as pd
    from aif360.datasets import BinaryLabelDataset

    df = pd.DataFrame({
        "x0": np.zeros(len(y)),  # dummy feature column (AIF360 expects >= 1 feature)
        SA: np.asarray(is_priv).astype(int),
        "label": np.asarray(y).astype(int),
    })
    return BinaryLabelDataset(
        df=df, label_names=["label"], protected_attribute_names=[SA],
        favorable_label=1, unfavorable_label=0,
    )


def metrics_aif360(y_true, y_pred, is_priv) -> dict:
    from aif360.metrics import ClassificationMetric

    ds_true = _aif360_dataset(y_true, is_priv)
    ds_pred = ds_true.copy(deepcopy=True)
    ds_pred.labels = np.asarray(y_pred).reshape(-1, 1).astype(float)
    m = ClassificationMetric(ds_true, ds_pred, unprivileged_groups=_UNPRIV, privileged_groups=_PRIV)
    return {
        "statistical_parity_difference": float(m.statistical_parity_difference()),
        "disparate_impact": float(m.disparate_impact()),
        "equal_opportunity_difference": float(m.equal_opportunity_difference()),
        "average_odds_difference": float(m.average_odds_difference()),
        "selection_rate_unprivileged": float(m.selection_rate(privileged=False)),
        "selection_rate_privileged": float(m.selection_rate(privileged=True)),
    }


def fairness_metrics(y_true, y_pred, is_priv, prefer_aif360: bool = True) -> dict:
    """Return the metrics dict, with a 'backend' key saying which library computed it."""
    global _AIF360_BROKEN
    if prefer_aif360 and not _AIF360_BROKEN:
        try:
            out = metrics_aif360(y_true, y_pred, is_priv)
            out["backend"] = "aif360"
            return out
        except Exception as exc:  # noqa: BLE001 - AIF360 0.6.x can break on very new NumPy/pandas
            _AIF360_BROKEN = True
            logger.warning("AIF360 unavailable or failed (%s: %s) - using the equivalent NumPy implementation.",
                           type(exc).__name__, exc)
    out = metrics_numpy(y_true, y_pred, is_priv)
    out["backend"] = "numpy"
    return out


# ------------------------------------------------------------------ mitigation
def reweighing_weights_numpy(y, is_priv) -> np.ndarray:
    """Kamiran & Calders reweighing: w(s, y) = P(s) * P(y) / P(s, y)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(is_priv).astype(int)
    n = len(y)
    w = np.ones(n, dtype=np.float64)
    for sv in (0, 1):
        for yv in (0, 1):
            mask = (s == sv) & (y == yv)
            if mask.sum() == 0:
                continue
            p_s = (s == sv).mean()
            p_y = (y == yv).mean()
            p_sy = mask.mean()
            w[mask] = p_s * p_y / p_sy
    return w


def reweighing_weights(y, is_priv, prefer_aif360: bool = True) -> np.ndarray:
    global _AIF360_BROKEN
    if prefer_aif360 and not _AIF360_BROKEN:
        try:
            from aif360.algorithms.preprocessing import Reweighing

            ds = _aif360_dataset(y, is_priv)
            rw = Reweighing(unprivileged_groups=_UNPRIV, privileged_groups=_PRIV)
            w = np.asarray(rw.fit_transform(ds).instance_weights, dtype=np.float64)
            if w.shape[0] == len(y) and np.all(np.isfinite(w)):
                return w
        except Exception as exc:  # noqa: BLE001
            _AIF360_BROKEN = True
            logger.warning("AIF360 Reweighing failed (%s: %s) - using the NumPy implementation.",
                           type(exc).__name__, exc)
    return reweighing_weights_numpy(y, is_priv)


# ------------------------------------------------------------------ report
def _improvement(before: float, after: float) -> float | None:
    if before is None or after is None or np.isnan(before) or abs(before) < 1e-12:
        return None
    return float((abs(before) - abs(after)) / abs(before) * 100.0)


def run() -> dict:
    """Compare every trained variant's test predictions -> fairness_report.json (+ PNG)."""
    report: dict = {"protected_attribute": SA, "privileged": cfg.PRIVILEGED_VALUE,
                    "unprivileged": cfg.UNPRIVILEGED_VALUE, "threshold": cfg.DECISION_THRESHOLD}
    found = []
    for variant in cfg.VARIANTS:
        npz = cfg.ARTIFACT_DIR / variant / "test_predictions.npz"
        if not npz.exists():
            continue
        d = np.load(npz)
        y_pred = (d["y_prob"] >= cfg.DECISION_THRESHOLD).astype(int)
        report[variant] = fairness_metrics(d["y_true"], y_pred, d["is_priv"])
        found.append(variant)
    if "baseline" not in found:
        raise FileNotFoundError("No baseline predictions found - run the training step first.")

    keys = ("statistical_parity_difference", "equal_opportunity_difference", "average_odds_difference")
    report["variants"] = found
    report["improvement_pct"] = {
        v: {k: _improvement(report["baseline"][k], report[v][k]) for k in keys} for v in found if v != "baseline"
    }
    out = cfg.ARTIFACT_DIR / "fairness_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    _plot(report)
    logger.info("Fairness report written to %s (backend: %s)", out, report["baseline"]["backend"])
    for v, d in report["improvement_pct"].items():
        logger.info("  %-12s |SPD| reduced by %s | |EOD| reduced by %s", v,
                    _fmt(d["statistical_parity_difference"]), _fmt(d["equal_opportunity_difference"]))
    return report


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.1f}%"


def _plot(report: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    keys = ["statistical_parity_difference", "equal_opportunity_difference", "average_odds_difference"]
    labels = ["Statistical\nparity diff.", "Equal opportunity\ndiff.", "Average odds\ndiff."]
    variants = report["variants"]
    x = np.arange(len(keys))
    w = 0.8 / len(variants)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, v in enumerate(variants):
        ax.bar(x + (i - (len(variants) - 1) / 2) * w, [abs(report[v][k]) for k in keys], w, label=v)
    ax.set_xticks(x, labels)
    ax.set_ylabel("|difference|  (0 = perfectly fair)")
    ax.set_title(f"Bias before vs after mitigation  (attribute: {SA})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.ARTIFACT_DIR / "fairness_comparison.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
