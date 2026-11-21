"""Generate artifacts/MODEL_CARD.md from the measured results (never hand-typed numbers)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fairml import config as cfg

logger = logging.getLogger(__name__)
MITIGATION_TEXT = {
    "none": "no mitigation (baseline)",
    "aif360_reweighing": "AIF360 Reweighing (pre-processing)",
    "adversarial_debiasing": "adversarial debiasing with a gradient-reversal layer (in-processing)",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _f(x, nd=3) -> str:
    return "n/a" if x is None or isinstance(x, str) or x != x else f"{x:.{nd}f}"


def run() -> Path:
    a = cfg.ARTIFACT_DIR
    meta = _load(cfg.MODEL_DIR / "metadata.json")
    fr, bench, shap_r = _load(a / "fairness_report.json"), _load(a / "benchmark.json"), _load(a / "shap_importance.json")
    champ, gate, tune = _load(a / "champion.json"), _load(a / "gate_report.json"), _load(a / "tuning" / "best_params.json")
    cal = _load(a / "calibration_report.json")
    variants = [v for v in cfg.VARIANTS if (a / v / "metrics.json").exists()]
    metrics = {v: _load(a / v / "metrics.json") for v in variants}
    prod = meta.get("variant", champ.get("champion", "?"))
    trained_on = meta.get("trained_on", {})

    L = [f"# Model Card - {cfg.REGISTERED_MODEL_NAME} v{cfg.MODEL_VERSION}", "",
         "## Model details",
         "- **Type:** feed-forward neural network (PyTorch), served through ONNX Runtime.",
         f"- **Deployed variant (champion):** `{prod}` - {MITIGATION_TEXT.get(meta.get('mitigation'), meta.get('mitigation'))}.",
         f"- **Why this variant:** {champ.get('reason', 'n/a')}.",
         f"- **Trained on:** {trained_on.get('gpu_name') or trained_on.get('device', 'n/a')} "
         f"(torch {trained_on.get('torch', 'n/a')}, CUDA build {trained_on.get('cuda_build')}).",
         f"- **Inputs:** {', '.join(cfg.FEATURES)}",
         f"- **Output:** probability that annual income exceeds 50K USD (threshold {cfg.DECISION_THRESHOLD}).", "",
         "## Intended use",
         "Educational / portfolio demonstration of a production-style, bias-audited ML workflow. "
         "**Not** intended for real decisions about people (hiring, lending, insurance, etc.).", "",
         "## Training data",
         "UCI Adult (Census Income), 1994 US census sample, ~48k rows, stratified 70/10/20 train/validation/test split. "
         "The data is old and reflects historical social biases.", "",
         "## Performance and fairness of all trained variants (held-out test set)",
         "| Variant | Accuracy | ROC-AUC | ECE | Stat. parity diff (0 ideal) | Disparate impact (1 ideal) | Equal opp. diff (0 ideal) |",
         "|---|---|---|---|---|---|---|"]
    for v in variants:
        m = metrics[v]
        L.append(f"| {'**' + v + '**' if v == prod else v} | {_f(m['performance']['accuracy'])} | {_f(m['performance']['roc_auc'])} | "
                 f"{_f(m.get('calibration', {}).get('ece'))} | {_f(m['fairness']['statistical_parity_difference'])} | "
                 f"{_f(m['fairness']['disparate_impact'])} | {_f(m['fairness']['equal_opportunity_difference'])} |")
    L += ["", f"Fairness metrics computed with: `{fr.get('baseline', {}).get('backend', 'n/a')}` "
          f"(protected attribute `{cfg.PROTECTED_ATTRIBUTE}`, privileged = {cfg.PRIVILEGED_VALUE}).", ""]
    for v, d in fr.get("improvement_pct", {}).items():
        L.append(f"- `{v}` vs baseline: |SPD| reduced by {_f(d.get('statistical_parity_difference'), 1)}%, "
                 f"|EOD| reduced by {_f(d.get('equal_opportunity_difference'), 1)}%.")

    if cal:
        L += ["", "## Calibration", "| Variant | Brier | ECE | ECE (men) | ECE (women) |", "|---|---|---|---|---|"]
        for v, c in cal.items():
            L.append(f"| {v} | {_f(c['brier'], 4)} | {_f(c['ece'])} | {_f(c['ece_privileged'])} | {_f(c['ece_unprivileged'])} |")

    if tune:
        L += ["", "## Hyper-parameter search (Optuna, multi-objective NSGA-II)",
              f"{tune['n_trials']} trials, Pareto front of {tune['pareto_size']} models. Chosen: variant `{tune['variant']}`, "
              f"hidden {tune['hidden']}, lr {tune['lr']:.2e}, dropout {tune['dropout']:.2f} "
              f"(val accuracy {_f(tune['val_accuracy'])}, |SPD| {_f(tune['val_abs_spd'])})."]

    if bench:
        L += ["", "## Serving performance",
              f"CPU engines pinned to 1 thread; GPU engines include host<->device copies. Device: "
              f"{bench.get('device', {}).get('gpu_name', 'CPU only')}.",
              "| Engine | batch=1 p50 (ms) | batch=256 rows/s | batch=4096 rows/s |", "|---|---|---|---|"]
        for n, d in bench["engines"].items():
            L.append(f"| {n} | {_f(d['batch_1']['p50_ms'], 4)} | {_f(d['batch_256']['throughput_rows_per_s'], 0)} | "
                     f"{_f(d['batch_4096']['throughput_rows_per_s'], 0)} |")
        ag = bench.get("accuracy_check", {}).get("int8_vs_fp32_label_agreement")
        if ag is not None:
            L.append(f"\nINT8 vs FP32 prediction agreement: {ag * 100:.2f}%. "
                     f"Quantised MatMul nodes: {meta.get('int8_info', {}).get('quantized_matmuls', 'n/a')}.")

    if shap_r:
        top = list(shap_r["mean_abs_shap"])[:5]
        L += ["", "## Explainability (SHAP)", f"Top global drivers: {', '.join(top)}. Protected attribute "
              f"`{cfg.PROTECTED_ATTRIBUTE}` ranks #{shap_r['protected_attribute_rank']} of {len(cfg.FEATURES)}."]

    if gate:
        L += ["", f"## Release gate: {'PASSED' if gate['passed'] else 'FAILED'}", "| Check | Value | Rule | Result |", "|---|---|---|---|"]
        for c in gate["checks"]:
            L.append(f"| {c['name']} | {c['value']} | {c['op']} {c['threshold']} | {'pass' if c['passed'] else 'FAIL'} |")

    L += ["", "## Monitoring",
          "The service exports Prometheus metrics: request rate/latency, predictions per group, live disparate impact, "
          "and per-feature PSI drift versus the training distribution (`/metrics`, `/drift`).",
          "", "## Ethical considerations & limitations",
          "- Gender is used as an input feature in this demo so mitigation effects are measurable; a real system must "
          "review whether using protected attributes is legal and ethical. `/counterfactual` shows the effect of flipping it.",
          "- Fairness is evaluated for one attribute (sex) and a few group definitions; race and intersectional groups are not mitigated.",
          "- Mitigation reduces, but does not eliminate, disparity and can trade off accuracy.",
          "- The dataset is from 1994; income distributions and social context have changed.", ""]
    out = a / "MODEL_CARD.md"
    out.write_text("\n".join(L))
    logger.info("Model card written to %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
