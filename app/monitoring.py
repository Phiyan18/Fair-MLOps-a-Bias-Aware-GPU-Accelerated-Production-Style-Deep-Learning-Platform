"""Prometheus-format metrics without extra dependencies."""
from __future__ import annotations

import threading
from collections import defaultdict


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: dict = defaultdict(int)          # (path, status) -> count
        self.duration_sum: dict = defaultdict(float)    # path -> seconds
        self.duration_count: dict = defaultdict(int)    # path -> count
        self.predictions: dict = defaultdict(int)       # (sex, label) -> count

    def observe_request(self, path: str, status: int, seconds: float) -> None:
        with self._lock:
            self.requests[(path, status)] += 1
            self.duration_sum[path] += seconds
            self.duration_count[path] += 1

    def observe_prediction(self, sex: str, label: str) -> None:
        with self._lock:
            self.predictions[(sex, label)] += 1

    def live_disparate_impact(self, unpriv: str, priv: str, positive: str, min_n: int = 20) -> float | None:
        """Selection-rate ratio of the unprivileged vs privileged group over all live predictions."""
        with self._lock:
            def rate(g):
                pos = self.predictions.get((g, positive), 0)
                tot = sum(c for (s, _), c in self.predictions.items() if s == g)
                return (pos / tot if tot >= min_n else None)
            ru, rp = rate(unpriv), rate(priv)
        return None if ru is None or rp in (None, 0) else ru / rp

    def render(self, drift: dict | None = None, live_di: float | None = None, model_info: dict | None = None) -> str:
        with self._lock:
            o = ["# HELP fairml_requests_total HTTP requests", "# TYPE fairml_requests_total counter"]
            o += [f'fairml_requests_total{{path="{p}",status="{s}"}} {c}' for (p, s), c in self.requests.items()]
            o += ["# HELP fairml_request_duration_seconds Request latency", "# TYPE fairml_request_duration_seconds summary"]
            for p in self.duration_sum:
                o.append(f'fairml_request_duration_seconds_sum{{path="{p}"}} {self.duration_sum[p]:.6f}')
                o.append(f'fairml_request_duration_seconds_count{{path="{p}"}} {self.duration_count[p]}')
            o += ["# HELP fairml_predictions_total Predictions by protected group and outcome",
                  "# TYPE fairml_predictions_total counter"]
            o += [f'fairml_predictions_total{{sex="{s}",label="{l}"}} {c}' for (s, l), c in self.predictions.items()]
        if live_di is not None:
            o += ["# HELP fairml_live_disparate_impact Live selection-rate ratio unprivileged/privileged (ideal 1, alert < 0.8)",
                  "# TYPE fairml_live_disparate_impact gauge", f"fairml_live_disparate_impact {live_di:.5f}"]
        if drift:
            o += ["# HELP fairml_drift_window_size Requests in the drift window", "# TYPE fairml_drift_window_size gauge",
                  f"fairml_drift_window_size {drift['window_size']}"]
            if drift["features"]:
                o += ["# HELP fairml_feature_psi Population Stability Index vs training data (alert >= 0.25)",
                      "# TYPE fairml_feature_psi gauge"]
                o += [f'fairml_feature_psi{{feature="{f}"}} {v["psi"]}' for f, v in drift["features"].items()]
        if model_info:
            o += ["# HELP fairml_model_info Model metadata", "# TYPE fairml_model_info gauge",
                  f'fairml_model_info{{version="{model_info["version"]}",variant="{model_info["variant"]}",'
                  f'file="{model_info["file"]}"}} 1']
        return "\n".join(o) + "\n"
