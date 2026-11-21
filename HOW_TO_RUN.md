# HOW TO RUN - Step-by-Step Installation & Execution Guide

Follow the steps **in order**. Run every command from the project root folder (the one containing this file).
Time needed: **about 3-4 hours** for everything (about 45 minutes for Steps 1-7).

Legend: 🐧 = Linux / macOS / WSL / Git-Bash, 🪟 = Windows PowerShell.

---

## STEP 0 - What you are building

```
Adult dataset -> [Optuna tuning] -> 3 PyTorch models on your GPU (baseline / reweighed / adversarial)
   -> MLflow -> fairness + calibration audit -> champion selection -> ONNX + INT8 -> benchmark (CPU vs GPU)
   -> SHAP -> release gate -> model card -> FastAPI (UI, explain, counterfactual, drift)
   -> Docker -> Prometheus + Grafana -> Kubernetes/OpenShift -> GitHub Actions (tests + security scans)
```

`python -m fairml.pipeline` does the whole ML part in one command. Everything after that is serving and deployment.

---

## STEP 1 - Install the prerequisites

| Tool | Needed for | Check with | Install |
|---|---|---|---|
| **Python 3.12 or 3.13** (3.11-3.14 work) | everything | `python --version` | https://www.python.org/downloads/ |
| **NVIDIA driver** (for your GTX 1650) | GPU training | `nvidia-smi` | https://www.nvidia.com/Download/index.aspx - **update to the newest driver** |
| **Git** | GitHub / CI | `git --version` | https://git-scm.com/downloads |
| **Docker Desktop** | Steps 9-10 | `docker --version` | https://docs.docker.com/get-docker/ |
| **kubectl** + **minikube** (or kind) | Step 11 | `kubectl version --client` | https://kubernetes.io/docs/tasks/tools/ |
| make (optional) | shortcuts | `make --version` | Windows: skip it, use the commands in this guide |

Steps 1-8 need only Python and the NVIDIA driver. Docker and Kubernetes can wait.

> You do **not** need to install the CUDA Toolkit or cuDNN separately: the PyTorch GPU wheel bundles what it needs.
> Only the NVIDIA **driver** has to be installed.

---

## STEP 2 - Unzip and create a virtual environment

```bash
# 🐧
unzip fair-mlops-ibm.zip && cd fair-mlops
python3 -m venv .venv
source .venv/bin/activate
```

```powershell
# 🪟  (right-click the zip -> Extract All, then open PowerShell inside the extracted fair-mlops folder)
python -m venv .venv
.venv\Scripts\Activate.ps1
# If activation is blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

The prompt should now start with `(.venv)`.

---

## STEP 3 - Install the latest libraries (with GPU PyTorch)

`requirements.txt` only lists *minimum* versions, so pip installs the **latest** releases (security fixes included).

### 3a. PyTorch with CUDA for the GTX 1650

The GTX 1650 (Turing, compute capability 7.5) is supported by current PyTorch CUDA builds. Look at the top-right of the
`nvidia-smi` output ("CUDA Version: xx.x") and choose:

```bash
python -m pip install --upgrade pip

# CUDA Version 13.x shown  (driver 580 or newer)  -> recommended:
pip install torch --index-url https://download.pytorch.org/whl/cu130

# CUDA Version 12.x shown (older driver): easiest is to UPDATE the driver and use cu130.
# Otherwise pick the matching build from https://pytorch.org/get-started/locally/ (e.g. cu128 / cu126)

# No NVIDIA GPU / want CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3b. Everything else

```bash
pip install -r requirements.txt
```

### 3c. Check your setup (important!)

```bash
python -m fairml.doctor
```

You want to see `[ OK ] CUDA usable: NVIDIA GeForce GTX 1650 ...` and `[ OK ] fp16 mixed precision (AMP) works`.
If it says you have the CPU-only PyTorch build or the GPU is not usable, it prints the exact fix.
**Everything still works on CPU** - the model is small - the GPU just makes training and the GPU benchmark faster.

### 3d. (Optional) ONNX Runtime on the GPU

Only for the extra "onnx_cuda" row in the benchmark. Needs a driver that supports CUDA 13:

```bash
pip uninstall -y onnxruntime
pip install "onnxruntime-gpu[cuda,cudnn]"
python -m fairml.doctor          # should list CUDAExecutionProvider
```

If it causes trouble, restore with `pip uninstall -y onnxruntime-gpu && pip install onnxruntime`. The project works either way.
(`onnxruntime` and `onnxruntime-gpu` must never be installed together.)

> Reproducibility tip after everything works: `pip freeze > requirements.lock.txt`

---

## STEP 4 - Run the tests (about 3-5 minutes)

```bash
pytest -q
```

All tests should pass. Some may be *skipped* (e.g. the AIF360 comparison if AIF360 does not support your NumPy/pandas yet).
The tests train tiny models on synthetic data in a temp folder, check the adversarial gradient-reversal maths, MLflow logging,
Optuna tuning, ONNX-vs-PyTorch parity, drift detection, the champion/gate logic, and every API endpoint including
API-key and rate-limit behaviour. They never touch your real `artifacts/` folder.

---

## STEP 5 - Run the full ML pipeline

```bash
python -m fairml.pipeline --tune 20
```

(`--tune 20` = 20 Optuna trials first, about 5-15 minutes. Drop it for a faster first run: `python -m fairml.pipeline`.)

What happens (the log shows `1/10 ... 10/10`):

1. **Optuna** searches hyper-parameters, optimising accuracy *and* fairness together -> Pareto front.
2. Trains **baseline**, **reweighed** (AIF360 pre-processing) and **adversarial** (in-processing) models **on your GPU**
   with mixed precision. Everything is logged to MLflow.
3. **Fairness report** (AIF360, or the equivalent NumPy implementation if AIF360 can't run on your versions).
4. **Calibration report** (are the probabilities trustworthy, also per gender?).
5. **Champion selection** - an auditable rule picks the variant to deploy.
6. **ONNX export + INT8 quantisation**, validated against PyTorch for several batch sizes.
7. **Benchmark**: PyTorch CPU / ONNX CPU / INT8 / PyTorch CUDA / (ONNX CUDA).
8. **SHAP** global explainability (slowest step: add `--skip-shap` to skip).
9. **Release gate** - go/no-go checks (accuracy, fairness, calibration, parity, INT8 agreement, latency).
10. **Model card** generated from the measured numbers.

Dataset: the UCI Adult dataset downloads automatically on the first run (OpenML, then UCI as fallback) and is cached in
`data/`. No internet? Use `--synthetic`. Or download
`https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data` manually and save it as `data/adult.data`.

Useful flags: `--device cpu`, `--no-amp` (disable fp16), `--epochs 15`, `--no-mlflow`, `--strict-gate` (exit with an error if the gate fails).

> **Do not panic if the release gate says FAILED.** It is a real check, not a formality. On the real dataset the baseline
> typically shows a large gender gap (disparate impact around 0.3) and mitigation closes much of it; whether a given variant
> clears the 0.8 rule depends on your run. Look at `artifacts/gate_report.json` and `artifacts/MODEL_CARD.md`, and explain the
> result honestly in your README - that is what an interviewer wants to see. You can tune thresholds with environment
> variables (e.g. `GATE_MAX_ABS_EOD=0.15`).

### Outputs (in `artifacts/`)

| File | What it is |
|---|---|
| `MODEL_CARD.md` | Auto-generated model card with *your measured* numbers |
| `fairness_report.json`, `fairness_comparison.png` | Bias of all 3 variants |
| `calibration_report.json`, `calibration.png` | Reliability diagram + ECE per group |
| `tuning/pareto_front.png`, `tuning/best_params.json` | Accuracy-vs-fairness trade-off from Optuna |
| `champion.json` | Which variant was deployed and why |
| `benchmark.json`, `benchmark.png` | Latency/throughput of every engine |
| `shap_importance.json/.png` | Global feature importance |
| `gate_report.json` | Release-gate checks |
| `model/` | The deployable bundle (ONNX, INT8, preprocessor, metadata, constraints) |

---

## STEP 6 - Look at the experiments in MLflow

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open http://localhost:5000 -> experiment **adult-income-fair-mlp** -> runs `baseline`, `reweighed`, `adversarial` (+ `tuning`).
Select all and click **Compare**. Screenshot the metrics (`test_accuracy`, `fair_*`, `calib_ece`, `gpu_peak_memory_mb`).
(MLflow 3.x stores runs in the SQLite file `mlflow.db`; the old `./mlruns` file store is deprecated.) Stop with Ctrl+C.

---

## STEP 7 - Run the API and the web UI

```bash
python -m uvicorn app.main:app --port 8000
```

- **http://localhost:8000** - the demo UI: edit an applicant, click **Predict**, **Explain** (feature contributions) or
  **Counterfactual fairness check** (flips *only* gender and shows how the prediction changes).
- **http://localhost:8000/docs** - interactive Swagger API docs.

Terminal checks (new terminal, same folder, venv active):

```bash
# 🐧
curl http://localhost:8000/ready
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @examples/predict_request.json
curl -X POST http://localhost:8000/explain -H "Content-Type: application/json" -d @examples/explain_request.json
curl -X POST http://localhost:8000/counterfactual -H "Content-Type: application/json" -d @examples/explain_request.json
curl http://localhost:8000/drift
curl http://localhost:8000/metrics
```

```powershell
# 🪟
Invoke-RestMethod http://localhost:8000/ready
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -InFile examples\predict_request.json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/counterfactual -ContentType "application/json" -InFile examples\explain_request.json
Invoke-RestMethod http://localhost:8000/drift
```

| Endpoint | Purpose |
|---|---|
| `GET /` | Demo web UI |
| `POST /predict` | Batch predictions |
| `POST /explain` | Which features pushed this prediction up or down |
| `POST /counterfactual` | Same person, only gender flipped - fairness sanity check |
| `GET /drift` | PSI data-drift report of live traffic vs training data |
| `GET /model-info` | Version, accuracy, calibration, fairness audit |
| `GET /health`, `GET /ready` | Kubernetes probes |
| `GET /metrics` | Prometheus metrics incl. live disparate impact and per-feature PSI |

Serve the INT8 model: `USE_QUANTIZED=1 python -m uvicorn ...` (🪟: `$env:USE_QUANTIZED="1"; python -m uvicorn ...`).

### Try the security features
```bash
# 🐧   require an API key + limit to 120 requests/minute/IP
API_KEY=my-secret RATE_LIMIT_PER_MIN=120 python -m uvicorn app.main:app --port 8000
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @examples/predict_request.json                       # 401
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -H "X-API-Key: my-secret" -d @examples/predict_request.json  # 200
```
```powershell
# 🪟
$env:API_KEY="my-secret"; $env:RATE_LIMIT_PER_MIN="120"; python -m uvicorn app.main:app --port 8000
```

### Try the drift detector (great for a demo video)
With the API running (and `--api-key my-secret` appended if you enabled a key):
```bash
python scripts/simulate_drift.py            # normal traffic  -> status ok
python scripts/simulate_drift.py --shift    # older people, longer hours -> PSI alarm on age / hours_per_week
```

### Load test
```bash
python scripts/load_test.py --url http://localhost:8000 --concurrency 16 --seconds 20
```

---

## STEP 8 - Monitoring stack: Prometheus + Grafana (Docker)

Needs Docker and the pipeline output from Step 5.

```bash
cp .env.example .env          # 🪟: copy .env.example .env     then edit GRAFANA_PASSWORD
docker compose up --build
```

| URL | What |
|---|---|
| http://localhost:8000 | The API + UI |
| http://localhost:9090 | Prometheus (try the *Alerts* tab) |
| http://localhost:3000 | Grafana (user `admin`, password from `.env`) -> Dashboards -> Fair-ML -> *Fair-ML service overview* |

Now run `python scripts/simulate_drift.py --shift` and `python scripts/load_test.py` and watch the dashboard:
request rate, latency, predictions per gender, **live disparate impact**, and **PSI drift bars** turn red; the
`FairMLDataDrift` alert fires in Prometheus. Screenshot it. Stop with `docker compose down`.

---

## STEP 9 - Build and run the Docker image on its own

```bash
docker build -t fairml-api:latest .
docker run --rm -p 8000:8000 --read-only --tmpfs /tmp --cap-drop ALL fairml-api:latest
```

If you trained with a different Python minor version, match it: `docker build --build-arg PYTHON_VERSION=3.12 -t fairml-api:latest .`
(the image installs the latest libraries but constrains numpy/pandas/scikit-learn to the exact versions that pickled the preprocessor).

---

## STEP 10 - Deploy to Kubernetes

### Option A: minikube (easiest)

```bash
minikube start
minikube addons enable metrics-server          # needed for autoscaling
minikube image load fairml-api:latest          # make the local image available in the cluster
kubectl apply -k k8s/
kubectl -n fairml rollout status deploy/fairml-api
kubectl -n fairml get pods,svc,hpa,pdb
kubectl -n fairml port-forward svc/fairml-api 8000:80
```

### Option B: kind

```bash
kind create cluster --name fairml
kind load docker-image fairml-api:latest --name fairml
kubectl apply -k k8s/
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl -n kube-system patch deploy metrics-server --type=json -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
kubectl -n fairml port-forward svc/fairml-api 8000:80
```

Test with Step 7's commands against `http://localhost:8000`.

**Autoscaling demo:** terminal 1 `kubectl -n fairml get hpa -w`; terminal 2
`python scripts/load_test.py --url http://localhost:8000 --concurrency 64 --seconds 120` - replicas grow from 2 toward 6.

**Optional hardening:** API key secret `kubectl -n fairml create secret generic fairml-secrets --from-literal=API_KEY="$(openssl rand -hex 24)"`
then `kubectl -n fairml rollout restart deploy/fairml-api`; network isolation `kubectl apply -f k8s/optional/networkpolicy.yaml`
(only enforced by CNIs that support it). Clean up: `kubectl delete -k k8s/`.

### Option C: Red Hat OpenShift Developer Sandbox (free; Red Hat is part of IBM)

1. Sign up at https://developers.redhat.com/developer-sandbox, log in to the console.
2. Top-right menu -> *Copy login command* -> run the `oc login ...` command.
3. Deploy into your own project using a registry image (push it via Step 11, make the package public):

```bash
NS=$(oc project -q)
IMG=ghcr.io/<your-github-user>/<your-repo>/fairml-api:latest
for f in configmap deployment service hpa pdb; do
  sed -e "s/namespace: fairml/namespace: $NS/" -e "s#image: fairml-api:latest#image: $IMG#" k8s/$f.yaml | oc apply -f -
done
oc apply -f k8s/openshift-route.yaml
oc get route fairml-api        # public HTTPS URL
```

---

## STEP 11 - GitHub + CI/CD + security scanning

```bash
git init
git add .
git commit -m "Fair-MLOps: GPU training, adversarial debiasing, Optuna, drift monitoring, Kubernetes"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

On every push the **Actions** tab runs: tests -> **bandit** (code security) -> **pip-audit** (vulnerable dependencies) ->
full pipeline (tune, train, audit, export, benchmark, gate) -> uploads model card + reports -> Docker build ->
**Trivy** image scan -> container smoke test -> push to GitHub Container Registry (main only).
**CodeQL** and **Dependabot** are configured too. For a run on the real dataset with the gate enforced:
Actions -> *CI-CD* -> *Run workflow* -> tick **real_data**.
(`artifacts/` is git-ignored; copy result images into `docs/images/` and commit those.)

---

## STEP 12 - Polish it so recruiters notice (do not skip!)

1. Fill the **README results tables** with *your measured numbers* (from `artifacts/MODEL_CARD.md`). Never quote numbers you did not measure.
2. Put in `docs/images/` and link from the README: `fairness_comparison.png`, `tuning/pareto_front.png`, `calibration.png`,
   `benchmark.png`, `shap_importance.png`, an MLflow *Compare* screenshot, the Grafana dashboard, and the web UI.
3. Record a **3-minute video**: `fairml.doctor` (GPU) -> pipeline log -> MLflow compare -> web UI counterfactual check ->
   drift simulation turning the Grafana panel red -> `kubectl get pods,hpa`.
4. GitHub: pin the repo; topics `mlops pytorch onnx kubernetes fairness aif360 mlflow fastapi optuna prometheus`.
5. Resume bullet (replace XX with measured values):
   *"Built a GPU-trained, bias-audited deep-learning platform: compared reweighing and adversarial debiasing (IBM AIF360 + PyTorch),
   automated champion selection and a fairness release gate, served via ONNX-INT8 on Kubernetes with Prometheus drift/fairness
   monitoring; cut the gender statistical-parity gap by XX% and CPU latency by XX%."*
6. Send the recruiter the repo link plus 3 lines of summary.

### Interview questions you should now be able to answer
- *Reweighing vs adversarial debiasing?* Reweighing changes sample weights (pre-processing, model-agnostic). Adversarial debiasing
  trains an adversary to predict gender from the hidden representation and reverses its gradient into the encoder (in-processing).
- *What does the gradient-reversal layer do?* Identity forward, multiplies the gradient by -lambda backward, so the encoder is pushed to *hide* the protected attribute.
- *Why a Pareto front?* Accuracy and fairness trade off; there is no single best model, so we expose the whole frontier and pick by an explicit constraint.
- *Why not trust accuracy alone?* Calibration (ECE) and per-group metrics catch a model that is accurate on average but unfair or over-confident for a group.
- *What is PSI and what do thresholds mean?* Population Stability Index compares live vs training distributions: <0.1 stable, 0.1-0.25 moderate, >0.25 major shift.
- *Why ONNX Runtime for serving?* Smaller image, fast CPU inference, no PyTorch at runtime. We validated numerical parity and measured INT8.
- *Why does the Docker image use a constraints file?* The preprocessor is a pickled scikit-learn object: it must be loaded with the same library versions that created it, while everything else stays on latest.
- *What would you do next?* Kubeflow/Argo pipelines, distributed training (DDP), Evidently reports, model signing/SBOM, watsonx.governance.

---

## TROUBLESHOOTING

| Problem | Fix |
|---|---|
| `python` not found (macOS/Linux) | Use `python3`. |
| `ModuleNotFoundError: fairml` | Run from the project root with the venv active. |
| doctor says "CPU-only PyTorch build" | `pip uninstall -y torch` then install the CUDA wheel from Step 3a. |
| doctor: "CUDA build installed but no usable GPU" | Update the NVIDIA driver, reboot, check `nvidia-smi`. Laptops: make sure Python uses the NVIDIA GPU (not only the integrated one). |
| `CUDA out of memory` | Not expected (the model uses ~100 MB of your 4 GB). Close other GPU apps or use `--device cpu`. |
| GPU slower than CPU in the benchmark | Normal for tiny batch sizes: copying data to the GPU costs more than the compute. The GPU wins at batch 4096. |
| AMP / fp16 problems on GTX 16xx | `--no-amp` (the GTX 1650 has no tensor cores, so AMP gives little benefit anyway). |
| `aif360` fails to install or log says "using the equivalent NumPy implementation" | AIF360 0.6.x can lag behind the newest NumPy/pandas. The NumPy implementation is mathematically identical; the model card states which backend was used. Optional: use a separate venv with older numpy/pandas to run real AIF360. |
| ONNX export error mentioning `onnxscript` | `pip install -U onnx onnxscript`. Or force the older exporter: `ONNX_EXPORTER=legacy python -m fairml.pipeline ...` (🪟 `$env:ONNX_EXPORTER="legacy"`). |
| Warning "INT8 model contains no quantised MatMul nodes" | The exporter fused MatMul+Add into Gemm. The project still works (INT8 just gives no speed-up); mention it in your write-up or use `ONNX_EXPORTER=legacy`. |
| MLflow error about the file store / `mlruns` | This project uses SQLite (`mlflow.db`). Use `--backend-store-uri sqlite:///mlflow.db`. |
| `Model bundle not found` when starting the API | Run `python -m fairml.pipeline` first. |
| Docker build: `artifacts/model/...` not found | The pipeline must run before `docker build`. |
| API logs: "scikit-learn version differs from training" | Rebuild the image after re-running the pipeline; don't mix a bundle with a different environment. |
| Pods `ImagePullBackOff` | `minikube image load fairml-api:latest` / `kind load docker-image fairml-api:latest`. |
| HPA shows `<unknown>/60%` | metrics-server not ready: wait 1-2 minutes (see Step 10). |
| Port already in use | Use another port, e.g. `--port 8001`. |
| 🪟 `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| A test fails | Run just that file: `pytest -q tests/test_api.py -x` and send me the last 30 lines. |
| Grafana shows "No data" | Send some traffic first (UI, `load_test.py`), wait ~10 s, check http://localhost:9090/targets is *UP*. |

If any step errors out, copy the **last 20-30 lines** of the error and ask for help.
