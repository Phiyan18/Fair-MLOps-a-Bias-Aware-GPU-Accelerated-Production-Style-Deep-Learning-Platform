"""Data-drift detection with the Population Stability Index (PSI). Pure NumPy, no heavy deps.

  PSI < 0.10         no significant shift
  0.10 <= PSI < 0.25 moderate shift   (warn)
  PSI >= 0.25        major shift      (alert -> investigate / retrain)

A reference distribution is computed at export time from the training data and shipped inside
the model bundle; the API compares a sliding window of live requests against it.
"""
from __future__ import annotations

import threading
from collections import deque

import numpy as np

WARN_PSI = 0.10
ALERT_PSI = 0.25
MIN_SAMPLES = 50
EPS = 1e-4


def _bin_props(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # right=True -> x <= edge goes to the lower bin, so a spike at 0 (capital_gain) is its own bin
    idx = np.digitize(x, edges, right=True)
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    return counts / max(counts.sum(), 1.0)


def build_reference(df, numeric: list[str], categorical: list[str], n_bins: int = 10) -> dict:
    ref: dict = {"n": int(len(df)), "numeric": {}, "categorical": {}}
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    for c in numeric:
        x = df[c].to_numpy(dtype=float)
        edges = np.unique(np.quantile(x, qs))
        ref["numeric"][c] = {
            "edges": [float(e) for e in edges],
            "proportions": [float(p) for p in _bin_props(x, edges)],
        }
    for c in categorical:
        vc = df[c].astype(str).value_counts(normalize=True)
        ref["categorical"][c] = {str(k): float(v) for k, v in vc.items()}
    return ref


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _status(value: float) -> str:
    return "alert" if value >= ALERT_PSI else "warn" if value >= WARN_PSI else "ok"


def drift_report(reference: dict, records: list[dict]) -> dict:
    n = len(records)
    out: dict = {"window_size": n, "min_samples": MIN_SAMPLES, "features": {}, "max_psi": None,
                 "status": "insufficient_data"}
    if n < MIN_SAMPLES or not reference:
        return out
    for c, spec in reference["numeric"].items():
        x = np.array([float(r[c]) for r in records])
        value = psi(np.array(spec["proportions"]), _bin_props(x, np.array(spec["edges"])))
        out["features"][c] = {"psi": round(value, 5), "status": _status(value)}
    for c, props in reference["categorical"].items():
        cats = list(props)
        pos = {k: i for i, k in enumerate(cats)}
        counts = np.zeros(len(cats) + 1)  # last bucket = categories never seen in training
        for r in records:
            counts[pos.get(str(r[c]), len(cats))] += 1
        expected = np.array([props[k] for k in cats] + [0.0])
        value = psi(expected, counts / counts.sum())
        out["features"][c] = {"psi": round(value, 5), "status": _status(value)}
    worst = max(v["psi"] for v in out["features"].values())
    out["max_psi"] = worst
    out["status"] = _status(worst)
    return out


class DriftMonitor:
    """Thread-safe sliding window of recent request records."""

    def __init__(self, window: int = 1000):
        self._buf: deque = deque(maxlen=window)
        self._lock = threading.Lock()

    def add(self, records: list[dict]) -> None:
        with self._lock:
            self._buf.extend(records)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self._buf)

    def report(self, reference: dict) -> dict:
        return drift_report(reference, self.snapshot())
