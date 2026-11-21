"""FastAPI service.

  GET  /                web demo UI            POST /predict         batch predictions
  GET  /docs            Swagger UI             POST /explain         per-request feature attribution
  GET  /health /ready   k8s probes             POST /counterfactual  flip the protected attribute, compare
  GET  /model-info      metadata + audit       GET  /drift           PSI data-drift report (live traffic vs training)
  GET  /metrics         Prometheus metrics (incl. live fairness + drift gauges)

Security: optional API key (API_KEY / API_KEYS), optional per-IP rate limit (RATE_LIMIT_PER_MIN),
strict security headers, request IDs, input validation with bounded sizes.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.monitoring import Metrics
from app.schemas import CounterfactualResponse, ExplainResponse, Person, PredictRequest, PredictResponse
from app.security import rate_limit, require_api_key
from fairml import config as cfg
from fairml.drift import DriftMonitor
from fairml.inference import Predictor

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fairml.api")

STATIC_DIR = Path(__file__).parent / "static"
state: dict = {"predictor": None, "drift": DriftMonitor(int(os.getenv("DRIFT_WINDOW", "1000")))}
metrics = Metrics()
PROTECTED = [Depends(rate_limit), Depends(require_api_key)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["predictor"] = Predictor()
        p = state["predictor"]
        logger.info("Model loaded: v%s variant=%s file=%s", p.version, p.metadata["variant"], p.model_file)
    except FileNotFoundError as exc:
        state["predictor"] = None
        logger.error("%s", exc)
    state["drift"] = DriftMonitor(int(os.getenv("DRIFT_WINDOW", "1000")))
    yield
    state["predictor"] = None


app = FastAPI(
    title="Fair-ML Income Classifier",
    description="Bias-audited neural network served with ONNX Runtime, with live drift and fairness monitoring.",
    version=cfg.MODEL_VERSION,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

UI_CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
          "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if not request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-store"
        return response
    finally:
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched" if status == 404 else request.url.path)
        metrics.observe_request(path, status, elapsed)
        if path not in ("/metrics", "/health", "/ready"):
            logger.info("rid=%s %s %s -> %s in %.1f ms", rid, request.method, path, status, elapsed * 1000)


def _clean(obj):
    """JSON cannot represent NaN/Infinity -> turn them into null."""
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _predictor() -> Predictor:
    p = state["predictor"]
    if p is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return p


# ------------------------------------------------------------------ UI + ops
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Content-Security-Policy": UI_CSP})


@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness probe: the process is up."""
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
def ready() -> dict:
    """Readiness probe: the model is loaded and the pod can receive traffic."""
    _predictor()
    return {"status": "ready"}


@app.get("/model-info", tags=["model"])
def model_info() -> dict:
    """Model metadata including accuracy, calibration and the fairness audit."""
    p = _predictor()
    m = p.metadata
    return _clean({
        "model_name": m["model_name"], "model_version": m["model_version"], "variant": m["variant"],
        "mitigation": m.get("mitigation"), "created_utc": m["created_utc"], "serving_file": p.model_file,
        "threshold": m["threshold"], "features": m["features"], "numeric_features": m["numeric_features"],
        "categorical_features": m["categorical_features"], "categories": m["categories"],
        "defaults": m["baselines"], "protected_attribute": m["protected_attribute"],
        "performance": m["performance"], "fairness": m["fairness"], "calibration": m.get("calibration"),
        "champion": m.get("champion"), "trained_on": m.get("trained_on"),
    })


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
def prometheus_metrics() -> Response:
    p = state["predictor"]
    drift = state["drift"].report(p.metadata.get("drift_reference", {})) if p else None
    di = metrics.live_disparate_impact(cfg.UNPRIVILEGED_VALUE, cfg.PRIVILEGED_VALUE, cfg.POSITIVE_LABEL)
    info = {"version": p.version, "variant": p.metadata["variant"], "file": p.model_file} if p else None
    return PlainTextResponse(metrics.render(drift, di, info), media_type="text/plain; version=0.0.4")


# ------------------------------------------------------------------ model endpoints
@app.post("/predict", response_model=PredictResponse, tags=["model"], dependencies=PROTECTED)
def predict(req: PredictRequest) -> PredictResponse:
    p = _predictor()
    start = time.perf_counter()
    records = [x.model_dump() for x in req.instances]
    preds = p.predict(records)
    for rec, pred in zip(records, preds):
        metrics.observe_prediction(rec["sex"], pred["label"])
    state["drift"].add(records)
    return PredictResponse(model_version=p.version, predictions=preds,
                           latency_ms=round((time.perf_counter() - start) * 1000, 3))


@app.post("/explain", response_model=ExplainResponse, tags=["model"], dependencies=PROTECTED)
def explain(person: Person) -> ExplainResponse:
    p = _predictor()
    return ExplainResponse(model_version=p.version, **p.explain(person.model_dump()))


@app.post("/counterfactual", response_model=CounterfactualResponse, tags=["fairness"], dependencies=PROTECTED)
def counterfactual(person: Person) -> CounterfactualResponse:
    """Counterfactual fairness check: identical person, only the protected attribute flipped."""
    p = _predictor()
    rec = person.model_dump()
    flipped = dict(rec, sex=cfg.UNPRIVILEGED_VALUE if rec["sex"] == cfg.PRIVILEGED_VALUE else cfg.PRIVILEGED_VALUE)
    a, b = p.predict([rec, flipped])
    delta = b["probability"] - a["probability"]
    changed = a["prediction"] != b["prediction"]
    if changed:
        text = "The decision CHANGES when only the protected attribute changes - a sign of direct dependence on it."
    elif abs(delta) >= 0.05:
        text = "The decision is the same but the probability moves noticeably - the model still uses the protected attribute."
    else:
        text = "Flipping the protected attribute barely changes the prediction for this person."
    return CounterfactualResponse(original=a, flipped=b, original_sex=rec["sex"], flipped_sex=flipped["sex"],
                                  probability_change=round(delta, 5), decision_changed=changed, interpretation=text)


@app.get("/drift", tags=["monitoring"], dependencies=PROTECTED)
def drift() -> dict:
    """PSI data-drift report: recent /predict traffic vs. the training distribution."""
    p = _predictor()
    return state["drift"].report(p.metadata.get("drift_reference", {}))
