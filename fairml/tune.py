"""Fairness-aware hyper-parameter search with Optuna (multi-objective, NSGA-II).

Two objectives are optimised at the same time:
    1. maximise validation accuracy
    2. minimise |statistical parity difference| on the validation set
There is no single "best" model - the result is a Pareto front (the accuracy/fairness trade-off curve).
We then pick the most accurate trial whose |SPD| is below --max-spd and write best_params.json,
which `python -m fairml.train --params-file ...` consumes.
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np

from fairml import config as cfg
from fairml import fairness
from fairml.device import resolve_device
from fairml.train import fit, predict_proba, prepare

logger = logging.getLogger(__name__)
ARCHS = {"s": (64, 32), "m": (128, 64), "l": (256, 128), "xl": (128, 64, 32)}


def run(*, n_trials: int = 20, synthetic: bool = False, n_synthetic: int = 6000, epochs: int = 12,
        seed: int = cfg.SEED, device: str = "auto", amp: bool = True, max_spd: float = 0.08,
        use_mlflow: bool = True) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dev = resolve_device(device)
    d = prepare(synthetic, n_synthetic, seed)
    w_rew = fairness.reweighing_weights(d.y_tr.astype(int), d.priv_tr).astype(np.float32)
    ones = np.ones(len(d.y_tr), dtype=np.float32)

    def objective(trial):
        variant = trial.suggest_categorical("variant", list(cfg.VARIANTS))
        arch = trial.suggest_categorical("arch", list(ARCHS))
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        bs = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
        adv_lambda = trial.suggest_float("adv_lambda", 0.2, 2.0) if variant == "adversarial" else 0.0
        model, _ = fit(d.X_tr, d.y_tr, w_rew if variant == "reweighed" else ones, d.X_va, d.y_va,
                       hidden=ARCHS[arch], dropout=dropout, epochs=epochs, batch_size=bs, lr=lr,
                       weight_decay=wd, patience=3, seed=seed + trial.number, device=dev, amp=amp,
                       s_tr=d.priv_tr if variant == "adversarial" else None, adv_lambda=adv_lambda or 1.0)
        p = predict_proba(model, d.X_va, dev)
        pred = (p >= cfg.DECISION_THRESHOLD).astype(int)
        acc = float((pred == d.y_va.astype(int)).mean())
        spd = abs(fairness.metrics_numpy(d.y_va.astype(int), pred, d.priv_va)["statistical_parity_difference"])
        trial.set_user_attr("val_accuracy", acc)
        trial.set_user_attr("val_abs_spd", float(spd))
        return acc, float(spd)

    def progress(study, trial):
        logger.info("trial %2d/%d  %-11s acc=%.4f |SPD|=%.4f", trial.number + 1, n_trials,
                    trial.params.get("variant", "?"), trial.values[0], trial.values[1])

    study = optuna.create_study(directions=["maximize", "minimize"], sampler=optuna.samplers.NSGAIISampler(seed=seed),
                                study_name="fair-mlp")
    study.optimize(objective, n_trials=n_trials, callbacks=[progress])

    pareto = study.best_trials
    ok = [t for t in pareto if t.values[1] <= max_spd]
    chosen = max(ok, key=lambda t: t.values[0]) if ok else min(pareto, key=lambda t: t.values[1])
    p = chosen.params
    best = {"variant": p["variant"], "hidden": list(ARCHS[p["arch"]]), "dropout": p["dropout"], "lr": p["lr"],
            "weight_decay": p["weight_decay"], "batch_size": p["batch_size"], "adv_lambda": p.get("adv_lambda", 0.0),
            "val_accuracy": chosen.values[0], "val_abs_spd": chosen.values[1], "n_trials": n_trials,
            "pareto_size": len(pareto), "constraint_max_spd": max_spd, "constraint_met": bool(ok)}

    out = cfg.ARTIFACT_DIR / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    (out / "best_params.json").write_text(json.dumps(best, indent=2))
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    _plot(study, pareto, chosen, out / "pareto_front.png")
    logger.info("Best trial: %s", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()}))

    if use_mlflow:
        try:
            from fairml.train import _setup_mlflow
            mlflow = _setup_mlflow()
            with mlflow.start_run(run_name="tuning", tags={"stage": "tuning", "project": "fair-mlops"}):
                mlflow.log_params({k: v for k, v in best.items() if k not in ("val_accuracy", "val_abs_spd")})
                for t in study.trials:
                    mlflow.log_metrics({"trial_val_accuracy": t.values[0], "trial_val_abs_spd": t.values[1]},
                                       step=t.number)
                mlflow.log_artifact(str(out / "pareto_front.png"))
                mlflow.log_artifact(str(out / "best_params.json"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow logging of the tuning run failed: %s", exc)
    return best


def _plot(study, pareto, chosen, path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"baseline": "#7f7f7f", "reweighed": "#1f77b4", "adversarial": "#ff7f0e"}
    for v, c in colors.items():
        pts = [(t.values[1], t.values[0]) for t in study.trials if t.params.get("variant") == v and t.values]
        if pts:
            ax.scatter(*zip(*pts), c=c, label=v, alpha=0.75)
    front = sorted((t.values[1], t.values[0]) for t in pareto)
    ax.step(*zip(*front), where="post", c="k", lw=1, label="Pareto front")
    ax.scatter([chosen.values[1]], [chosen.values[0]], s=200, facecolors="none", edgecolors="r", lw=2, label="chosen")
    ax.set_xlabel("|statistical parity difference|  (lower = fairer)")
    ax.set_ylabel("validation accuracy  (higher = better)")
    ax.set_title("Accuracy vs fairness trade-off (Optuna, NSGA-II)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-spd", type=float, default=0.08)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(n_trials=a.trials, epochs=a.epochs, synthetic=a.synthetic, max_spd=a.max_spd, device=a.device,
        amp=not a.no_amp, use_mlflow=not a.no_mlflow)


if __name__ == "__main__":
    main()
