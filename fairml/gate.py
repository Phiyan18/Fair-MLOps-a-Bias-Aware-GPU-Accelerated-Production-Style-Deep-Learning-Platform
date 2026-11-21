"""Champion selection + release quality gate.

select_champion():  decides which trained variant gets deployed (data-driven, auditable).
release_gate():     final go / no-go checks on the exported bundle (accuracy, fairness, calibration,
                    ONNX parity, INT8 agreement, latency). Run in CI with `python -m fairml.gate --fail`.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

from fairml import config as cfg

logger = logging.getLogger(__name__)


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _fair_ok(m: dict) -> bool:
    f = m["fairness"]
    di, eod = f.get("disparate_impact"), f.get("equal_opportunity_difference")
    if _nan(di) or _nan(eod):
        return False
    return cfg.GATE_MIN_DI <= di <= cfg.GATE_MAX_DI and abs(eod) <= cfg.GATE_MAX_ABS_EOD


def _gap(m: dict) -> float:
    f = m["fairness"]
    return (abs(f["statistical_parity_difference"]) + abs(f["equal_opportunity_difference"])) / 2


def select_champion(explicit: str | None = None) -> dict:
    """Pick the variant to deploy and write artifacts/champion.json."""
    metrics = {v: _load(cfg.ARTIFACT_DIR / v / "metrics.json") for v in cfg.VARIANTS}
    metrics = {v: m for v, m in metrics.items() if m}
    if not metrics:
        raise FileNotFoundError("No trained variants found - run the training step first.")

    explicit = explicit or cfg.PRODUCTION_VARIANT
    if explicit and explicit != "auto":
        if explicit not in metrics:
            raise ValueError(f"Variant '{explicit}' was not trained. Available: {list(metrics)}")
        champion, reason = explicit, "forced via PRODUCTION_VARIANT / argument"
    else:
        passing = [v for v, m in metrics.items() if _fair_ok(m)]
        if passing:
            champion = max(passing, key=lambda v: metrics[v]["performance"]["roc_auc"])
            reason = (f"highest ROC-AUC among variants meeting the fairness criteria "
                      f"(DI in [{cfg.GATE_MIN_DI}, {cfg.GATE_MAX_DI}], |EOD| <= {cfg.GATE_MAX_ABS_EOD})")
        else:
            base_acc = metrics.get("baseline", next(iter(metrics.values())))["performance"]["accuracy"]
            eligible = [v for v, m in metrics.items()
                        if m["performance"]["accuracy"] >= base_acc - cfg.GATE_MAX_ACC_DROP] or list(metrics)
            champion = min(eligible, key=lambda v: _gap(metrics[v]))
            reason = (f"no variant met the fairness criteria; chose the smallest fairness gap among variants "
                      f"losing <= {cfg.GATE_MAX_ACC_DROP:.0%} accuracy vs baseline")

    out = {
        "champion": champion, "reason": reason,
        "candidates": {v: {"accuracy": m["performance"]["accuracy"], "roc_auc": m["performance"]["roc_auc"],
                           "disparate_impact": m["fairness"]["disparate_impact"],
                           "abs_spd": abs(m["fairness"]["statistical_parity_difference"]),
                           "abs_eod": abs(m["fairness"]["equal_opportunity_difference"]),
                           "meets_fairness_criteria": _fair_ok(m)} for v, m in metrics.items()},
    }
    (cfg.ARTIFACT_DIR / "champion.json").write_text(json.dumps(out, indent=2))
    logger.info("Champion model: %s  (%s)", champion, reason)
    return out


def champion_variant() -> str:
    c = _load(cfg.ARTIFACT_DIR / "champion.json")
    return c.get("champion") or (cfg.PRODUCTION_VARIANT if cfg.PRODUCTION_VARIANT != "auto" else "reweighed")


def _check(name, value, op, threshold) -> dict:
    ok = False if _nan(value) else {">=": value >= threshold, "<=": value <= threshold}[op]
    return {"name": name, "value": None if _nan(value) else round(float(value), 5), "op": op,
            "threshold": threshold, "passed": bool(ok)}


def release_gate(fail: bool = False) -> dict:
    variant = champion_variant()
    m = _load(cfg.ARTIFACT_DIR / variant / "metrics.json")
    meta = _load(cfg.MODEL_DIR / "metadata.json")
    bench = _load(cfg.ARTIFACT_DIR / "benchmark.json")
    synthetic = bool(m.get("params", {}).get("synthetic"))
    min_acc, min_auc = (0.70, 0.70) if synthetic else (cfg.GATE_MIN_ACCURACY, cfg.GATE_MIN_AUC)

    checks = [
        _check("accuracy", m["performance"]["accuracy"], ">=", min_acc),
        _check("roc_auc", m["performance"]["roc_auc"], ">=", min_auc),
        _check("disparate_impact >= 0.8 (80% rule)", m["fairness"]["disparate_impact"], ">=", cfg.GATE_MIN_DI),
        _check("disparate_impact <= 1.25", m["fairness"]["disparate_impact"], "<=", cfg.GATE_MAX_DI),
        _check("|equal_opportunity_difference|", abs(m["fairness"]["equal_opportunity_difference"]), "<=",
               cfg.GATE_MAX_ABS_EOD),
        _check("calibration ECE", m.get("calibration", {}).get("ece"), "<=", cfg.GATE_MAX_ECE),
    ]
    pc = meta.get("parity_checks", {})
    if "max_abs_diff_onnx_vs_torch" in pc:
        checks.append(_check("ONNX vs PyTorch max |diff|", pc["max_abs_diff_onnx_vs_torch"], "<=", 1e-4))
    if "int8_label_agreement" in pc:
        checks.append(_check("INT8 vs FP32 label agreement", pc["int8_label_agreement"], ">=",
                             cfg.GATE_MIN_INT8_AGREEMENT))
    onnx_cpu = bench.get("engines", {}).get("onnx_cpu_fp32", {}).get("batch_1")
    if onnx_cpu:
        checks.append(_check("ONNX CPU p95 latency (batch 1, ms)", onnx_cpu["p95_ms"], "<=", 50.0))

    passed = all(c["passed"] for c in checks)
    report = {"variant": variant, "passed": passed, "synthetic_data_relaxed_thresholds": synthetic, "checks": checks}
    (cfg.ARTIFACT_DIR / "gate_report.json").write_text(json.dumps(report, indent=2))
    logger.info("Release gate for '%s': %s", variant, "PASSED" if passed else "FAILED")
    for c in checks:
        logger.info("  [%s] %-40s %s %s %s", "ok" if c["passed"] else "!!", c["name"], c["value"], c["op"], c["threshold"])
    if fail and not passed:
        sys.exit(1)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fail", action="store_true", help="exit with status 1 if the gate does not pass")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    release_gate(fail=a.fail)
