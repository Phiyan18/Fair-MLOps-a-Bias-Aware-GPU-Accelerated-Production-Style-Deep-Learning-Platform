import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_model_shapes_and_adversary():
    from fairml.model import Adversary, TabularMLP

    m = TabularMLP(20, (16, 8), 0.1)
    x = torch.randn(5, 20)
    assert m(x).shape == (5, 1)
    adv = Adversary(m.out_features)
    z = m.encoder(x)
    out = adv(z, 1.0)
    out.sum().backward()
    # gradient reversal: encoder receives gradients that come from the adversary
    assert m.encoder[0].weight.grad is not None


def test_gradient_reversal_flips_sign():
    from fairml.model import _GradReverse

    x = torch.ones(3, requires_grad=True)
    _GradReverse.apply(x, 2.0).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -2.0))


def test_fit_improves_and_adversary_trains():
    from fairml.train import fit, predict_proba

    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 10)).astype(np.float32)
    s = rng.random(600) < 0.5
    y = ((X[:, 0] + 0.5 * s + rng.normal(scale=0.3, size=600)) > 0.2).astype(np.float32)
    w = np.ones(600, dtype=np.float32)
    kw = dict(hidden=(16, 8), dropout=0.1, epochs=8, batch_size=64, lr=5e-3, weight_decay=1e-4, patience=8, seed=0)
    for s_tr in (None, s):
        model, hist = fit(X[:500], y[:500], w[:500], X[500:], y[500:], s_tr=None if s_tr is None else s_tr[:500], **kw)
        assert hist[-1]["train_loss"] < hist[0]["train_loss"]
        p = predict_proba(model, X[500:])
        assert ((p >= 0.5) == y[500:]).mean() > 0.7


def test_device_resolution_falls_back():
    from fairml.device import resolve_device

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda"}


def test_mlflow_logging_with_sqlite_backend(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    import mlflow

    from fairml import config as cfg
    from fairml import train

    monkeypatch.setattr(cfg, "ARTIFACT_DIR", tmp_path)   # keep the session artifacts untouched

    train.run(synthetic=True, n_synthetic=1500, epochs=2, hidden=(8,), device="cpu", variants=("baseline",),
              use_mlflow=True)
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(cfg.EXPERIMENT_NAME)
    runs = mlflow.search_runs([exp.experiment_id])
    assert len(runs) >= 1 and "metrics.test_accuracy" in runs.columns


def test_tuning_smoke(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    from fairml import config as cfg
    from fairml import tune

    monkeypatch.setattr(cfg, "ARTIFACT_DIR", tmp_path)

    best = tune.run(n_trials=3, synthetic=True, n_synthetic=1500, epochs=2, device="cpu", use_mlflow=False)
    assert best["variant"] in {"baseline", "reweighed", "adversarial"}
    assert 0 <= best["val_accuracy"] <= 1
