import pytest

PERSON = {"age": 39, "education_num": 13, "capital_gain": 2174, "capital_loss": 0, "hours_per_week": 40,
          "workclass": "State-gov", "marital_status": "Never-married", "occupation": "Adm-clerical",
          "relationship": "Not-in-family", "race": "White", "sex": "Male", "native_country": "United-States"}


@pytest.fixture()
def client(artifacts):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:  # context manager runs the lifespan (loads the model)
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}


def test_ui_and_security_headers(client):
    r = client.get("/")
    assert r.status_code == 200 and "Fair-ML" in r.text
    assert "script-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert client.get("/static/app.js").status_code == 200


def test_predict_single_and_batch(client):
    r = client.post("/predict", json={"instances": [PERSON, dict(PERSON, sex="Female")]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 2
    assert all(0 <= p["probability"] <= 1 for p in body["predictions"])
    assert "x-request-id" in r.headers


@pytest.mark.parametrize("bad", [dict(PERSON, age=-5), dict(PERSON, sex="Other"),
                                 {k: v for k, v in PERSON.items() if k != "workclass"}])
def test_predict_validation_errors(client, bad):
    assert client.post("/predict", json={"instances": [bad]}).status_code == 422


def test_empty_batch_rejected(client):
    assert client.post("/predict", json={"instances": []}).status_code == 422


def test_explain(client):
    body = client.post("/explain", json=PERSON).json()
    assert len(body["contributions"]) == 12 and "base_probability" in body


def test_counterfactual(client):
    body = client.post("/counterfactual", json=PERSON).json()
    assert body["original_sex"] == "Male" and body["flipped_sex"] == "Female"
    assert body["probability_change"] == pytest.approx(body["flipped"]["probability"] - body["original"]["probability"], abs=1e-4)


def test_model_info(client):
    body = client.get("/model-info").json()
    assert "fairness" in body and "categories" in body and "defaults" in body


def test_metrics_and_drift(client):
    for _ in range(3):
        client.post("/predict", json={"instances": [PERSON] * 30})
    text = client.get("/metrics").text
    assert "fairml_requests_total" in text and 'fairml_predictions_total{sex="Male"' in text
    assert "fairml_drift_window_size 90" in text and "fairml_feature_psi" in text
    rep = client.get("/drift").json()
    assert rep["window_size"] == 90 and rep["status"] in {"ok", "warn", "alert"}


def test_api_key_enforced(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    assert client.post("/predict", json={"instances": [PERSON]}).status_code == 401
    assert client.post("/predict", json={"instances": [PERSON]}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/predict", json={"instances": [PERSON]}, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200      # probes stay open


def test_rate_limit(client, monkeypatch):
    from app.security import limiter

    limiter.reset()
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "3")
    codes = [client.post("/predict", json={"instances": [PERSON]}).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    limiter.reset()
