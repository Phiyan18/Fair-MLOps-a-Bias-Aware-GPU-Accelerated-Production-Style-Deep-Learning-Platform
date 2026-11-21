# Security notes

## Supply chain
- `requirements.txt` lists only *minimum* versions, so installs pick the **latest** releases (with security fixes).
- Dependabot opens weekly PRs; CI runs **pip-audit** (known CVEs), **bandit** (static analysis), **CodeQL**, and a
  **Trivy** scan of the container image.
- The Docker image installs the latest libraries but is *constrained* to the exact numpy / pandas / scikit-learn
  versions that created the pickled preprocessor (`artifacts/model/constraints.txt`).
  Pickled scikit-learn objects must only be loaded from sources you trust - never load `preprocessor.joblib` from strangers.

## Runtime hardening
- Container runs as a non-root user (UID 10001), read-only root file system, all Linux capabilities dropped,
  `no-new-privileges`, seccomp `RuntimeDefault` (Kubernetes), no service-account token mounted.
- Optional API key (`API_KEY` / `API_KEYS`, constant-time comparison) and optional per-IP rate limit (`RATE_LIMIT_PER_MIN`).
- Strict input validation (bounded sizes/ranges), security headers, a strict Content-Security-Policy on the demo UI
  (no inline scripts, DOM updates via `textContent` only), request IDs in logs.
- Optional Kubernetes NetworkPolicy (`k8s/optional/networkpolicy.yaml`) and PodDisruptionBudget.
- Local compose stack binds all ports to 127.0.0.1 only.

## Not included (be aware)
TLS termination (use an Ingress / OpenShift Route), authentication beyond a shared API key, secret management
(use Kubernetes Secrets / a vault), per-user quotas, model signing. This is a demonstration project.

## Reporting
Open a private security advisory on the GitHub repository.
