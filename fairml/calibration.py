"""Calibration metrics (are predicted probabilities trustworthy?) - overall and per group."""
from __future__ import annotations

import json
import logging

import numpy as np

from fairml import config as cfg

logger = logging.getLogger(__name__)


def ece(y_true, y_prob, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    if len(p) == 0:
        return float("nan")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total, err = len(p), 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            err += m.sum() / total * abs(y[m].mean() - p[m].mean())
    return float(err)


def brier(y_true, y_prob) -> float:
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_prob, dtype=float)
    return float(np.mean((p - y) ** 2))


def reliability_curve(y_true, y_prob, n_bins: int = 10):
    """Returns (mean predicted prob, observed frequency, count) for each non-empty bin."""
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_prob, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    conf, acc, cnt = [], [], []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            conf.append(p[m].mean())
            acc.append(y[m].mean())
            cnt.append(int(m.sum()))
    return np.array(conf), np.array(acc), np.array(cnt)


def summary(y_true, y_prob, is_priv) -> dict:
    y, p, s = np.asarray(y_true), np.asarray(y_prob), np.asarray(is_priv).astype(bool)
    return {
        "brier": brier(y, p),
        "ece": ece(y, p),
        "ece_privileged": ece(y[s], p[s]),
        "ece_unprivileged": ece(y[~s], p[~s]),
    }


def run() -> dict:
    """Reliability diagram + JSON report for every trained variant."""
    report = {}
    curves = {}
    for variant in cfg.VARIANTS:
        npz = cfg.ARTIFACT_DIR / variant / "test_predictions.npz"
        if not npz.exists():
            continue
        d = np.load(npz)
        report[variant] = summary(d["y_true"], d["y_prob"], d["is_priv"])
        curves[variant] = reliability_curve(d["y_true"], d["y_prob"])
    (cfg.ARTIFACT_DIR / "calibration_report.json").write_text(json.dumps(report, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
        for v, (c, a, _) in curves.items():
            ax.plot(c, a, marker="o", label=f"{v} (ECE {report[v]['ece']:.3f})")
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed frequency of >50K")
        ax.set_title("Reliability diagram (test set)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(cfg.ARTIFACT_DIR / "calibration.png", dpi=140)
        plt.close(fig)
    except Exception:  # noqa: BLE001
        pass
    logger.info("Calibration report written (%s)", ", ".join(f"{v}: ECE={r['ece']:.3f}" for v, r in report.items()))
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
