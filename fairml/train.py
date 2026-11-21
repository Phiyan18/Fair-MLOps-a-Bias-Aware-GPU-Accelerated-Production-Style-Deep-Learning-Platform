"""Train three model variants and log everything to MLflow.

  baseline     plain training
  reweighed    pre-processing mitigation: AIF360 reweighing -> weighted loss
  adversarial  in-processing mitigation: an adversary tries to recover `sex` from the hidden
               representation; a gradient-reversal layer pushes the encoder to hide it

GPU notes: the whole (small) dataset is moved to the GPU once and batched there (no DataLoader
overhead); mixed precision (AMP, fp16) is used on CUDA unless --no-amp is given.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn

from fairml import calibration, fairness
from fairml import config as cfg
from fairml import data as data_mod
from fairml.device import device_info, resolve_device
from fairml.model import Adversary, TabularMLP

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------ data preparation (shared with tune.py)
@dataclass
class Prepared:
    train_df: object
    val_df: object
    test_df: object
    pre: object
    X_tr: np.ndarray
    X_va: np.ndarray
    X_te: np.ndarray
    y_tr: np.ndarray
    y_va: np.ndarray
    y_te: np.ndarray
    priv_tr: np.ndarray
    priv_va: np.ndarray
    priv_te: np.ndarray


def prepare(synthetic: bool = False, n_synthetic: int = 6000, seed: int = cfg.SEED) -> Prepared:
    df = data_mod.make_synthetic(n_synthetic, seed) if synthetic else data_mod.load_adult()
    logger.info("Dataset: %d rows, %.1f%% positive class", len(df), 100 * df[cfg.TARGET].mean())
    train_df, val_df, test_df = data_mod.split_data(df, seed)
    pre = data_mod.build_preprocessor().fit(train_df[cfg.FEATURES])

    def x(d):
        return pre.transform(d[cfg.FEATURES]).astype(np.float32)

    def priv(d):
        return (d[cfg.PROTECTED_ATTRIBUTE] == cfg.PRIVILEGED_VALUE).to_numpy()

    return Prepared(train_df, val_df, test_df, pre, x(train_df), x(val_df), x(test_df),
                    train_df[cfg.TARGET].to_numpy().astype(np.float32),
                    val_df[cfg.TARGET].to_numpy().astype(np.float32),
                    test_df[cfg.TARGET].to_numpy().astype(int),
                    priv(train_df), priv(val_df), priv(test_df))


# ------------------------------------------------------------------ model fitting
@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: str = "cpu", batch_size: int = 16384) -> np.ndarray:
    model = model.to(device).eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        out.append(torch.sigmoid(model(xb)).squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)


def fit(X_tr, y_tr, w_tr, X_va, y_va, *, hidden, dropout, epochs, batch_size, lr, weight_decay,
        patience, seed, device="cpu", amp=False, s_tr=None, adv_lambda=1.0, on_epoch=None):
    """Weighted-BCE training loop with early stopping (+ optional adversary). Returns (model_on_cpu, history)."""
    set_seed(seed)
    dev = torch.device(device)
    use_amp = bool(amp and dev.type == "cuda")
    model = TabularMLP(X_tr.shape[1], tuple(hidden), dropout).to(dev)
    adversary = Adversary(model.out_features).to(dev) if s_tr is not None else None
    params = list(model.parameters()) + (list(adversary.parameters()) if adversary is not None else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    Xt = torch.from_numpy(X_tr).to(dev)
    yt = torch.from_numpy(y_tr).to(dev)
    wt = torch.from_numpy(np.asarray(w_tr, dtype=np.float32)).to(dev)
    st = torch.from_numpy(np.asarray(s_tr, dtype=np.float32)).to(dev) if adversary is not None else None
    Xv, yv = torch.from_numpy(X_va).to(dev), torch.from_numpy(y_va).to(dev)
    n = len(Xt)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    total_steps = max(1, epochs * math.ceil(n / batch_size))
    step = 0

    best_loss, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        if adversary is not None:
            adversary.train()
        perm = torch.randperm(n, device=dev, generator=gen)
        running = torch.zeros((), device=dev)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb, wb = Xt[idx], yt[idx], wt[idx]
            # ramp the adversary strength 0 -> adv_lambda (stabilises adversarial training)
            lam = adv_lambda * (2.0 / (1.0 + math.exp(-10.0 * step / total_steps)) - 1.0)
            step += 1
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                z = model.encoder(xb)
                logits = model.head(z).squeeze(-1)
                loss_clf = (bce(logits.float(), yb) * wb).sum() / wb.sum()
                loss = loss_clf
                if adversary is not None:
                    adv_logits = adversary(z, lam).squeeze(-1)
                    loss = loss_clf + bce(adv_logits.float(), st[idx]).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss_clf.detach() * len(idx)
        train_loss = running.item() / n

        model.eval()
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            vlog = model(Xv).squeeze(-1).float()
        val_loss = nn.functional.binary_cross_entropy_with_logits(vlog, yv).item()
        val_acc = ((vlog > 0).float() == yv).float().mean().item()
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc}
        history.append(row)
        logger.info("epoch %02d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f", epoch, train_loss, val_loss, val_acc)
        if on_epoch:
            on_epoch(row)

        if val_loss < best_loss - 1e-4:
            best_loss, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                logger.info("early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), history


def evaluate(y_true, y_prob, is_priv) -> dict:
    y_pred = (y_prob >= cfg.DECISION_THRESHOLD).astype(int)
    perf = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }
    return {"performance": perf, "fairness": fairness.fairness_metrics(y_true, y_pred, is_priv),
            "calibration": calibration.summary(y_true, y_prob, is_priv)}


# ------------------------------------------------------------------ MLflow helpers
def _setup_mlflow():
    import mlflow
    from mlflow import MlflowClient

    uri = cfg.MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    if not uri.startswith("http"):  # local backends: decide where artifacts are written
        client = MlflowClient()
        if client.get_experiment_by_name(cfg.EXPERIMENT_NAME) is None:
            client.create_experiment(cfg.EXPERIMENT_NAME, artifact_location=cfg.MLFLOW_ARTIFACT_ROOT)
    mlflow.set_experiment(cfg.EXPERIMENT_NAME)
    return mlflow


@contextlib.contextmanager
def _mlflow_run(enabled: bool, run_name: str, tags: dict):
    if not enabled:
        yield None
        return
    mlflow = _setup_mlflow()
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        yield mlflow
        logger.info("MLflow run id: %s", run.info.run_id)


def _log_model(mlflow, model, variant: str) -> None:
    import mlflow.pytorch

    name = f"{cfg.REGISTERED_MODEL_NAME}-{variant}"
    try:
        mlflow.pytorch.log_model(model, name="model", registered_model_name=name)  # MLflow 3.x
    except TypeError:
        mlflow.pytorch.log_model(model, artifact_path="model", registered_model_name=name)  # MLflow 2.x


def load_params_file(path) -> dict:
    p = json.loads(Path(path).read_text())
    return {k: p[k] for k in ("hidden", "dropout", "lr", "weight_decay", "batch_size", "adv_lambda") if k in p}


# ------------------------------------------------------------------ main entry
def run(*, synthetic: bool = False, n_synthetic: int = 6000, epochs: int = 30, batch_size: int = 256,
        lr: float = 1e-3, hidden: tuple[int, ...] = (128, 64), dropout: float = 0.2,
        weight_decay: float = 1e-4, patience: int = 5, seed: int = cfg.SEED, adv_lambda: float = 1.0,
        device: str = "auto", amp: bool = True, params_file: str | None = None,
        variants: tuple[str, ...] = cfg.VARIANTS, use_mlflow: bool = True) -> dict:
    if params_file:  # e.g. artifacts/tuning/best_params.json produced by fairml.tune
        o = load_params_file(params_file)
        hidden = tuple(o.get("hidden", hidden))
        dropout, lr = o.get("dropout", dropout), o.get("lr", lr)
        weight_decay, batch_size = o.get("weight_decay", weight_decay), int(o.get("batch_size", batch_size))
        adv_lambda = o.get("adv_lambda", adv_lambda) or adv_lambda
        logger.info("Using tuned hyper-parameters from %s: %s", params_file, o)

    dev = resolve_device(device)
    info = device_info(dev)
    logger.info("Training device: %s%s", dev, f" ({info.get('gpu_name')}, {info.get('vram_gb')} GB)" if dev == "cuda" else "")
    set_seed(seed)

    d = prepare(synthetic, n_synthetic, seed)
    cfg.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(d.pre, cfg.ARTIFACT_DIR / "preprocessor.joblib")
    splits = cfg.ARTIFACT_DIR / "splits"
    splits.mkdir(exist_ok=True)
    for name, df in (("train", d.train_df), ("val", d.val_df), ("test", d.test_df)):
        df.to_csv(splits / f"{name}.csv", index=False)
    (cfg.ARTIFACT_DIR / "baselines.json").write_text(json.dumps(data_mod.feature_baselines(d.train_df), indent=2))

    ones = np.ones(len(d.y_tr), dtype=np.float32)
    weights = {"baseline": ones, "adversarial": ones,
               "reweighed": fairness.reweighing_weights(d.y_tr.astype(int), d.priv_tr).astype(np.float32)}
    logger.info("Reweighing weights range: %.3f - %.3f", weights["reweighed"].min(), weights["reweighed"].max())
    mitigation = {"baseline": "none", "reweighed": "aif360_reweighing", "adversarial": "adversarial_debiasing"}

    results = {}
    for variant in variants:
        logger.info("=== Training variant: %s ===", variant)
        params = dict(variant=variant, epochs=epochs, batch_size=batch_size, lr=lr, hidden=list(hidden),
                      dropout=dropout, weight_decay=weight_decay, patience=patience, seed=seed,
                      n_train=len(d.y_tr), n_features=d.X_tr.shape[1], synthetic=synthetic,
                      mitigation=mitigation[variant], device=dev, amp=bool(amp and dev == "cuda"),
                      adv_lambda=adv_lambda if variant == "adversarial" else 0.0)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        tags = {"variant": variant, "project": "fair-mlops", "device": dev, "gpu": info.get("gpu_name", "none")}
        with _mlflow_run(use_mlflow, variant, tags) as mlf:
            if mlf:
                mlf.log_params(params)
            t0 = time.time()
            model, history = fit(
                d.X_tr, d.y_tr, weights[variant], d.X_va, d.y_va, hidden=hidden, dropout=dropout, epochs=epochs,
                batch_size=batch_size, lr=lr, weight_decay=weight_decay, patience=patience, seed=seed,
                device=dev, amp=amp, s_tr=d.priv_tr if variant == "adversarial" else None, adv_lambda=adv_lambda,
                on_epoch=(lambda r: mlf.log_metrics({k: v for k, v in r.items() if k != "epoch"}, step=r["epoch"]))
                if mlf else None,
            )
            train_seconds = time.time() - t0

            y_prob = predict_proba(model, d.X_te, dev)
            model = model.cpu()
            res = evaluate(d.y_te, y_prob, d.priv_te)
            res.update(variant=variant, train_seconds=train_seconds, epochs_run=len(history), params=params,
                       device=info)
            if dev == "cuda":
                res["gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
            results[variant] = res
            logger.info("%s -> %s", variant, json.dumps({k: round(v, 4) for k, v in res["performance"].items()}))
            logger.info("%s -> %s", variant, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                          for k, v in res["fairness"].items()}))

            vdir = cfg.ARTIFACT_DIR / variant
            vdir.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, vdir / "model.pt")
            (vdir / "model_config.json").write_text(json.dumps(
                {"in_features": int(d.X_tr.shape[1]), "hidden": list(hidden), "dropout": dropout}))
            (vdir / "metrics.json").write_text(json.dumps(res, indent=2))
            np.savez(vdir / "test_predictions.npz", y_true=d.y_te, y_prob=y_prob, is_priv=d.priv_te)

            if mlf:
                mlf.log_metrics({f"test_{k}": v for k, v in res["performance"].items()})
                mlf.log_metrics({f"calib_{k}": v for k, v in res["calibration"].items()})
                mlf.log_metrics({f"fair_{k}": v for k, v in res["fairness"].items()
                                 if isinstance(v, float) and not np.isnan(v)})
                mlf.log_metric("train_seconds", train_seconds)
                if "gpu_peak_memory_mb" in res:
                    mlf.log_metric("gpu_peak_memory_mb", res["gpu_peak_memory_mb"])
                mlf.log_artifact(str(vdir / "metrics.json"))
                try:
                    _log_model(mlf, model, variant)
                except Exception as exc:  # noqa: BLE001 - registry is optional
                    logger.warning("Could not log/register model in MLflow: %s", exc)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="use generated data (works offline)")
    ap.add_argument("--n-synthetic", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--adv-lambda", type=float, default=1.0)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--no-amp", action="store_true", help="disable fp16 mixed precision on GPU")
    ap.add_argument("--params-file", help="JSON with tuned hyper-parameters (from fairml.tune)")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(synthetic=a.synthetic, n_synthetic=a.n_synthetic, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        hidden=tuple(a.hidden), dropout=a.dropout, adv_lambda=a.adv_lambda, device=a.device, amp=not a.no_amp,
        params_file=a.params_file, use_mlflow=not a.no_mlflow)


if __name__ == "__main__":
    main()
