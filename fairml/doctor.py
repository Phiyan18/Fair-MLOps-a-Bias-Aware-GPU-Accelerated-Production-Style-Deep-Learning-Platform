"""Environment check:  python -m fairml.doctor

Tells you in plain language whether your GPU (e.g. GTX 1650) is usable and which libraries are installed.
"""
from __future__ import annotations

import importlib.metadata as md
import platform
import shutil
import subprocess
import sys

OK, WARN, BAD = "[ OK ]", "[WARN]", "[FAIL]"
CORE = ["numpy", "pandas", "scikit-learn", "torch", "onnx", "onnxruntime", "mlflow", "fastapi", "pydantic"]
OPTIONAL = ["onnxscript", "onnxruntime-gpu", "aif360", "shap", "optuna", "uvicorn", "pytest", "httpx"]


def _ver(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def main() -> int:
    problems = 0
    print(f"Python {platform.python_version()} on {platform.platform()}")
    if sys.version_info < (3, 11):
        print(f"{BAD} Python 3.11+ is required (3.12 or 3.13 recommended).")
        problems += 1

    print("\nLibraries")
    for pkg in CORE:
        v = _ver(pkg)
        print(f"  {OK if v else BAD} {pkg:<16} {v or 'NOT INSTALLED'}")
        problems += v is None
    for pkg in OPTIONAL:
        v = _ver(pkg)
        print(f"  {OK if v else WARN} {pkg:<16} {v or 'not installed (optional)'}")

    print("\nGPU")
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            print(f"  {OK} nvidia-smi: {out}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {WARN} nvidia-smi failed: {exc}")
    else:
        print(f"  {WARN} nvidia-smi not found - install the NVIDIA driver (https://www.nvidia.com/Download/index.aspx)")

    try:
        import torch

        print(f"  torch {torch.__version__} | CUDA build: {torch.version.cuda} | cuDNN: {torch.backends.cudnn.version()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"  {OK} CUDA usable: {p.name} | compute capability {p.major}.{p.minor} | {p.total_memory / 1024 ** 3:.1f} GB VRAM")
            x = torch.randn(1024, 1024, device="cuda")
            torch.cuda.synchronize()
            print(f"  {OK} fp32 matmul on GPU works (checksum {float((x @ x).sum()):.1f})")
            try:
                with torch.autocast("cuda", dtype=torch.float16):
                    y = x @ x
                torch.cuda.synchronize()
                print(f"  {OK} fp16 mixed precision (AMP) works, result dtype {y.dtype}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {WARN} AMP test failed ({exc}); train with --no-amp")
        else:
            problems += 0
            if torch.version.cuda is None:
                print(f"  {WARN} You installed the CPU-only PyTorch build. For your GPU run:\n"
                      "        pip uninstall -y torch\n"
                      "        pip install torch --index-url https://download.pytorch.org/whl/cu130\n"
                      "        (needs NVIDIA driver >= 580; for older drivers use .../whl/cu126, or update the driver)")
            else:
                print(f"  {WARN} CUDA build installed but no usable GPU: update the NVIDIA driver and re-run nvidia-smi.")
    except ImportError:
        print(f"  {BAD} torch is not installed")
        problems += 1

    print("\nONNX Runtime")
    try:
        import onnxruntime as ort

        print(f"  providers: {ort.get_available_providers()}")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            print(f"  {WARN} ONNX Runtime GPU not active (optional). CPU inference is used.\n"
                  '        Optional: pip uninstall -y onnxruntime && pip install "onnxruntime-gpu[cuda,cudnn]"')
    except ImportError:
        print(f"  {BAD} onnxruntime is not installed")
        problems += 1

    print("\nResult:", "everything needed is installed." if not problems else f"{problems} problem(s) found - see FAIL lines.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
