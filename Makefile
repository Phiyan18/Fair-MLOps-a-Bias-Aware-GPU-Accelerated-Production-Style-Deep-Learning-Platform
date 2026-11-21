# Convenience targets (Linux / macOS / WSL / Git-Bash). On plain Windows use the commands in HOW_TO_RUN.md.
PY ?= python

.PHONY: install doctor test pipeline pipeline-tune pipeline-offline mlflow-ui serve docker-build up down k8s-deploy k8s-delete load-test drift-demo audit clean

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

doctor:             ## check GPU / libraries
	$(PY) -m fairml.doctor

test:
	$(PY) -m pytest -q

pipeline:           ## real Adult dataset, GPU if available
	$(PY) -m fairml.pipeline

pipeline-tune:      ## + 25-trial Optuna search (accuracy vs fairness)
	$(PY) -m fairml.pipeline --tune 25

pipeline-offline:   ## generated data, no internet
	$(PY) -m fairml.pipeline --synthetic --epochs 10

mlflow-ui:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

serve:
	$(PY) -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

docker-build:
	docker build -t fairml-api:latest .

up:                 ## API + Prometheus + Grafana
	docker compose up --build

down:
	docker compose down

k8s-deploy:
	kubectl apply -k k8s/
	kubectl -n fairml rollout status deploy/fairml-api

k8s-delete:
	kubectl delete -k k8s/

load-test:
	$(PY) scripts/load_test.py --url http://localhost:8000 --concurrency 32 --seconds 60

drift-demo:
	$(PY) scripts/simulate_drift.py && $(PY) scripts/simulate_drift.py --shift

audit:              ## security checks
	$(PY) -m pip_audit
	$(PY) -m bandit -q -r fairml app

clean:
	rm -rf artifacts/baseline artifacts/reweighed artifacts/adversarial artifacts/model artifacts/splits artifacts/tuning artifacts/*.json artifacts/*.png artifacts/*.npz artifacts/*.joblib artifacts/MODEL_CARD.md mlflow.db mlartifacts .pytest_cache
