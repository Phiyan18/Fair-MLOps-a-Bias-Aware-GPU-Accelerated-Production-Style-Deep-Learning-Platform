"""Run the whole workflow with one command:

    python -m fairml.pipeline                         # real Adult dataset, GPU if available
    python -m fairml.pipeline --tune 25               # + Optuna fairness-aware hyper-parameter search first
    python -m fairml.pipeline --synthetic --epochs 8  # offline smoke-run with generated data
    python -m fairml.pipeline --device cpu --no-amp

Steps: [tune] -> train 3 variants -> fairness report -> calibration -> pick champion -> ONNX/INT8 export ->
       benchmark (CPU + GPU) -> SHAP -> release gate -> model card.
"""
from __future__ import annotations

import argparse
import logging
import time

from fairml import benchmark, calibration, explain, export_onnx, fairness, gate, model_card, train, tune
from fairml import config as cfg

logger = logging.getLogger("pipeline")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="generated data, works offline")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tune", type=int, default=0, metavar="N", help="run N Optuna trials before training")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--no-amp", action="store_true", help="disable fp16 mixed precision on GPU")
    ap.add_argument("--skip-shap", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--strict-gate", action="store_true", help="exit with an error if the release gate fails")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    t0 = time.time()
    params_file = None
    gate_result: dict = {}
    steps = []
    if a.tune > 0:
        def do_tune():
            nonlocal params_file
            tune.run(n_trials=a.tune, synthetic=a.synthetic, device=a.device, amp=not a.no_amp,
                     use_mlflow=not a.no_mlflow)
            params_file = str(cfg.ARTIFACT_DIR / "tuning" / "best_params.json")
        steps.append(("Hyper-parameter search (Optuna, accuracy vs fairness)", do_tune))
    steps += [
        ("Train baseline + reweighed + adversarial models", lambda: train.run(
            synthetic=a.synthetic, epochs=a.epochs, device=a.device, amp=not a.no_amp, params_file=params_file,
            use_mlflow=not a.no_mlflow)),
        ("Fairness report (AIF360 / NumPy)", fairness.run),
        ("Calibration report", calibration.run),
        ("Select champion model", gate.select_champion),
        ("Export champion to ONNX + INT8", export_onnx.run),
        ("Benchmark CPU / GPU engines", benchmark.run),
        ("SHAP explainability", (lambda: None) if a.skip_shap else explain.run),
        ("Release quality gate", lambda: gate_result.update(gate.release_gate(fail=False))),
        ("Generate model card", model_card.run),
    ]
    for i, (name, fn) in enumerate(steps, 1):
        logger.info("\n########## %d/%d  %s ##########", i, len(steps), name)
        fn()
    logger.info("\nAll done in %.1fs. Outputs: ./artifacts (deployable bundle: ./artifacts/model)", time.time() - t0)
    if a.strict_gate and not gate_result.get("passed", False):
        raise SystemExit("Release gate FAILED (see artifacts/gate_report.json)")


if __name__ == "__main__":
    main()
