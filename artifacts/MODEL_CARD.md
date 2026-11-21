# Model Card - adult-income-classifier v2.0.0

## Model details
- **Type:** feed-forward neural network (PyTorch), served through ONNX Runtime.
- **Deployed variant (champion):** `adversarial` - adversarial debiasing with a gradient-reversal layer (in-processing).
- **Why this variant:** no variant met the fairness criteria; chose the smallest fairness gap among variants losing <= 5% accuracy vs baseline.
- **Trained on:** NVIDIA GeForce GTX 1650 (torch 2.14.1+cu130, CUDA build 13.0).
- **Inputs:** age, education_num, capital_gain, capital_loss, hours_per_week, workclass, marital_status, occupation, relationship, race, sex, native_country
- **Output:** probability that annual income exceeds 50K USD (threshold 0.5).

## Intended use
Educational / portfolio demonstration of a production-style, bias-audited ML workflow. **Not** intended for real decisions about people (hiring, lending, insurance, etc.).

## Training data
UCI Adult (Census Income), 1994 US census sample, ~48k rows, stratified 70/10/20 train/validation/test split. The data is old and reflects historical social biases.

## Performance and fairness of all trained variants (held-out test set)
| Variant | Accuracy | ROC-AUC | ECE | Stat. parity diff (0 ideal) | Disparate impact (1 ideal) | Equal opp. diff (0 ideal) |
|---|---|---|---|---|---|---|
| baseline | 0.857 | 0.915 | 0.012 | -0.165 | 0.325 | -0.072 |
| reweighed | 0.851 | 0.908 | 0.015 | -0.069 | 0.632 | 0.175 |
| **adversarial** | 0.852 | 0.908 | 0.011 | -0.118 | 0.475 | 0.057 |

Fairness metrics computed with: `aif360` (protected attribute `sex`, privileged = Male).

- `reweighed` vs baseline: |SPD| reduced by 58.0%, |EOD| reduced by -141.9%.
- `adversarial` vs baseline: |SPD| reduced by 28.6%, |EOD| reduced by 20.9%.

## Calibration
| Variant | Brier | ECE | ECE (men) | ECE (women) |
|---|---|---|---|---|
| baseline | 0.0974 | 0.012 | 0.014 | 0.011 |
| reweighed | 0.1022 | 0.015 | 0.047 | 0.050 |
| adversarial | 0.1010 | 0.011 | 0.021 | 0.042 |

## Hyper-parameter search (Optuna, multi-objective NSGA-II)
30 trials, Pareto front of 9 models. Chosen: variant `reweighed`, hidden [64, 32], lr 2.90e-03, dropout 0.40 (val accuracy 0.850, |SPD| 0.079).

## Serving performance
CPU engines pinned to 1 thread; GPU engines include host<->device copies. Device: NVIDIA GeForce GTX 1650.
| Engine | batch=1 p50 (ms) | batch=256 rows/s | batch=4096 rows/s |
|---|---|---|---|
| pytorch_cpu_fp32 | 0.0821 | 2010792 | 4353464 |
| onnx_cpu_fp32 | 0.0139 | 4109017 | 5065441 |
| onnx_cpu_int8 | 0.0196 | 3611687 | 4983979 |
| pytorch_cuda_fp32 | 0.3365 | 630983 | 1553239 |
| onnx_cuda_fp32 | 0.1544 | 1346228 | 3167541 |

INT8 vs FP32 prediction agreement: 99.63%. Quantised MatMul nodes: 3.

## Explainability (SHAP)
Top global drivers: marital_status, education_num, age, relationship, occupation. Protected attribute `sex` ranks #10 of 12.

## Release gate: FAILED
| Check | Value | Rule | Result |
|---|---|---|---|
| accuracy | 0.85239 | >= 0.8 | pass |
| roc_auc | 0.90816 | >= 0.85 | pass |
| disparate_impact >= 0.8 (80% rule) | 0.47477 | >= 0.8 | FAIL |
| disparate_impact <= 1.25 | 0.47477 | <= 1.25 | pass |
| |equal_opportunity_difference| | 0.05706 | <= 0.1 | pass |
| calibration ECE | 0.01137 | <= 0.1 | pass |
| ONNX vs PyTorch max |diff| | 0.0 | <= 0.0001 | pass |
| INT8 vs FP32 label agreement | 0.9975 | >= 0.98 | pass |
| ONNX CPU p95 latency (batch 1, ms) | 0.0177 | <= 50.0 | pass |

## Monitoring
The service exports Prometheus metrics: request rate/latency, predictions per group, live disparate impact, and per-feature PSI drift versus the training distribution (`/metrics`, `/drift`).

## Ethical considerations & limitations
- Gender is used as an input feature in this demo so mitigation effects are measurable; a real system must review whether using protected attributes is legal and ethical. `/counterfactual` shows the effect of flipping it.
- Fairness is evaluated for one attribute (sex) and a few group definitions; race and intersectional groups are not mitigated.
- Mitigation reduces, but does not eliminate, disparity and can trade off accuracy.
- The dataset is from 1994; income distributions and social context have changed.
