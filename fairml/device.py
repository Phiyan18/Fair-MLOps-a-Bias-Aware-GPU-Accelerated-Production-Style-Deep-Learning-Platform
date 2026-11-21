"""GPU / CPU device helpers (training side)."""
from __future__ import annotations

import logging
import platform

logger = logging.getLogger(__name__)


def resolve_device(pref: str = "auto") -> str:
    """'auto' -> 'cuda' when a CUDA GPU is usable, else 'cpu'."""
    import torch

    pref = (pref or "auto").lower()
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available (torch %s, CUDA build: %s) - using CPU. "
                       "Run `python -m fairml.doctor` to diagnose.", torch.__version__, torch.version.cuda)
        return "cpu"
    return pref


def device_info(device: str | None = None) -> dict:
    import torch

    avail = torch.cuda.is_available()
    info = {
        "device": device or ("cuda" if avail else "cpu"),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": avail,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if avail:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        info.update(gpu_name=props.name, compute_capability=f"{props.major}.{props.minor}",
                    vram_gb=round(props.total_memory / 1024 ** 3, 2))
    return info
