"""Latency / throughput benchmark of every available inference engine.

CPU engines are pinned to ONE thread (apples-to-apples). GPU engines are included automatically when a
CUDA GPU / onnxruntime-gpu is available and measured END-TO-END (host->device copy + compute + copy back).
"""
from __future__ import annotations

import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
import torch

from fairml import config as cfg
from fairml.device import device_info
from fairml.export_onnx import _load_torch
from fairml.gate import champion_variant

logger = logging.getLogger(__name__)
BATCH_SIZES = (1, 32, 256, 4096)
BASELINE_ENGINE = "pytorch_cpu_fp32"


def _time_it(fn, n_iter: int, warmup: int = 15) -> np.ndarray:
    for _ in range(warmup):
        fn()
    times = np.empty(n_iter)
    for i in range(n_iter):
        t0 = time.perf_counter()
        fn()
        times[i] = (time.perf_counter() - t0) * 1000.0
    return times


def _stats(times_ms: np.ndarray, batch: int) -> dict:
    return {"p50_ms": float(np.percentile(times_ms, 50)), "p95_ms": float(np.percentile(times_ms, 95)),
            "p99_ms": float(np.percentile(times_ms, 99)), "mean_ms": float(times_ms.mean()),
            "throughput_rows_per_s": float(batch / (times_ms.mean() / 1000.0))}


def _take(X: np.ndarray, n: int) -> np.ndarray:
    reps = int(np.ceil(n / len(X)))
    return np.ascontiguousarray(np.tile(X, (reps, 1))[:n])


def _ort_session(path, providers, threads: int | None = 1):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), sess_options=opts, providers=providers)


def build_engines(variant: str) -> dict:
    import onnxruntime as ort

    torch.set_num_threads(1)
    engines = {}
    model_cpu = _load_torch(variant)

    def torch_cpu(x):
        with torch.no_grad():
            return torch.sigmoid(model_cpu(torch.from_numpy(x))).numpy()

    engines["pytorch_cpu_fp32"] = torch_cpu
    for name, fname in (("onnx_cpu_fp32", "model.onnx"), ("onnx_cpu_int8", "model_int8.onnx")):
        path = cfg.MODEL_DIR / fname
        if path.exists():
            sess = _ort_session(path, ["CPUExecutionProvider"])
            inp = sess.get_inputs()[0].name
            engines[name] = (lambda s, i: (lambda x: s.run(None, {i: x})[0]))(sess, inp)

    if torch.cuda.is_available():
        model_gpu = _load_torch(variant).to("cuda")

        def torch_cuda(x):
            with torch.no_grad():
                return torch.sigmoid(model_gpu(torch.from_numpy(x).to("cuda"))).cpu().numpy()

        engines["pytorch_cuda_fp32"] = torch_cuda
    if "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            sess = _ort_session(cfg.MODEL_DIR / "model.onnx", ["CUDAExecutionProvider", "CPUExecutionProvider"], None)
            if sess.get_providers()[0] == "CUDAExecutionProvider":
                inp = sess.get_inputs()[0].name
                engines["onnx_cuda_fp32"] = lambda x: sess.run(None, {inp: x})[0]
            else:
                logger.warning("onnxruntime-gpu installed but CUDA provider could not be initialised (CUDA/cuDNN libs?).")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ONNX Runtime CUDA engine unavailable: %s", exc)
    return engines


def run(variant: str | None = None, n_iter: int = 300) -> dict:
    variant = variant or champion_variant()
    pre = joblib.load(cfg.ARTIFACT_DIR / "preprocessor.joblib")
    test_df = pd.read_csv(cfg.ARTIFACT_DIR / "splits" / "test.csv")
    X_all = pre.transform(test_df[cfg.FEATURES]).astype(np.float32)
    y_all = test_df[cfg.TARGET].to_numpy()

    engines = build_engines(variant)
    results: dict = {"variant": variant, "cpu_threads": 1, "iterations": n_iter, "device": device_info(),
                     "note": "CUDA engines include host<->device transfer", "engines": {}}
    for name, fn in engines.items():
        results["engines"][name] = {}
        for bs in BATCH_SIZES:
            x = _take(X_all, bs)
            iters = n_iter if bs == 1 else max(30, n_iter // (3 if bs < 1000 else 10))
            results["engines"][name][f"batch_{bs}"] = _stats(_time_it(lambda: fn(x), iters), bs)
        e = results["engines"][name]
        logger.info("%-18s batch=1 p50=%.3f ms | batch=4096 %.0f rows/s", name, e["batch_1"]["p50_ms"],
                    e["batch_4096"]["throughput_rows_per_s"])

    base = results["engines"][BASELINE_ENGINE]
    results["speedup_vs_pytorch_cpu"] = {
        name: {b: base[b]["p50_ms"] / data[b]["p50_ms"] for b in data}
        for name, data in results["engines"].items() if name != BASELINE_ENGINE}

    acc = {}
    probs = {}
    for name, fn in engines.items():
        if name.startswith("onnx_cpu"):
            probs[name] = np.asarray(fn(X_all)).reshape(-1)
            acc[name] = {"accuracy": float(np.mean((probs[name] >= cfg.DECISION_THRESHOLD) == y_all))}
    if "onnx_cpu_int8" in probs and "onnx_cpu_fp32" in probs:
        acc["int8_vs_fp32_label_agreement"] = float(np.mean((probs["onnx_cpu_int8"] >= 0.5) == (probs["onnx_cpu_fp32"] >= 0.5)))
    results["accuracy_check"] = acc
    results["model_size_kb"] = {f: (cfg.MODEL_DIR / f).stat().st_size / 1024
                                for f in ("model.onnx", "model_int8.onnx") if (cfg.MODEL_DIR / f).exists()}

    (cfg.ARTIFACT_DIR / "benchmark.json").write_text(json.dumps(results, indent=2))
    _plot(results)
    logger.info("Benchmark written to %s", cfg.ARTIFACT_DIR / "benchmark.json")
    return results


def _plot(results: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    names = list(results["engines"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        axes[0].bar(np.arange(len(BATCH_SIZES)) + i * w, [results["engines"][n][f"batch_{b}"]["p50_ms"] for b in BATCH_SIZES], w, label=n)
        axes[1].bar(np.arange(len(BATCH_SIZES)) + i * w, [results["engines"][n][f"batch_{b}"]["throughput_rows_per_s"] for b in BATCH_SIZES], w, label=n)
    for ax, title in zip(axes, ("Median latency (ms) - lower is better", "Throughput (rows/s) - higher is better")):
        ax.set_xticks(np.arange(len(BATCH_SIZES)) + w * (len(names) - 1) / 2, [f"batch {b}" for b in BATCH_SIZES])
        ax.set_title(title, fontsize=10)
        ax.set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg.ARTIFACT_DIR / "benchmark.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
