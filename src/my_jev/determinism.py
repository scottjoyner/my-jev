"""Make a training run reproducible, or admit that it is not.

Seeding the RNGs is not enough on CUDA. Reduction and scatter kernels pick
their accumulation order at runtime, so the same seed can still produce a
different model — and when that variation is as wide as the effect being
measured, every comparison drawn from single runs is noise.

`configure_determinism` pins the algorithms that decide that order. It also
reports whether determinism actually took effect, so a caller can record the
truth instead of assuming it.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any
CUBLAS_WORKSPACE_CONFIG_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


@dataclass
class DeterminismReport:
    seed: int
    device: str
    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cublas_workspace_config: str
    warnings: list[str] = field(default_factory=list)

    @property
    def reproducible(self) -> bool:
        return not self.warnings

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "device": self.device,
            "deterministic_algorithms": self.deterministic_algorithms,
            "cudnn_deterministic": self.cudnn_deterministic,
            "cudnn_benchmark": self.cudnn_benchmark,
            "cublas_workspace_config": self.cublas_workspace_config,
            "reproducible": self.reproducible,
            "warnings": list(self.warnings),
        }


def configure_determinism(seed: int) -> DeterminismReport:
    """Seed every RNG and pin the kernels whose order varies at runtime.

    `CUBLAS_WORKSPACE_CONFIG` has to be set before CUDA initialises, so it is
    written here and verified; a run that starts with the variable already
    imported torch is reported as non-reproducible rather than quietly trusted.
    """
    os.environ.setdefault(CUBLAS_WORKSPACE_CONFIG_ENV, CUBLAS_WORKSPACE_CONFIG)
    warnings: list[str] = []
    if os.environ.get(CUBLAS_WORKSPACE_CONFIG_ENV) != CUBLAS_WORKSPACE_CONFIG:
        warnings.append(
            f"{CUBLAS_WORKSPACE_CONFIG_ENV} is "
            f"{os.environ.get(CUBLAS_WORKSPACE_CONFIG_ENV)!r}, expected "
            f"{CUBLAS_WORKSPACE_CONFIG!r}"
        )

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        warnings.append("numpy is unavailable; its RNG is unseeded")

    import torch

    torch.manual_seed(seed)
    device = "cpu"
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as error:  # pragma: no cover - depends on the build
        warnings.append(f"use_deterministic_algorithms failed: {error}")

    return DeterminismReport(
        seed=seed,
        device=device,
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
        cudnn_deterministic=bool(getattr(torch.backends.cudnn, "deterministic", False)),
        cudnn_benchmark=bool(getattr(torch.backends.cudnn, "benchmark", False)),
        cublas_workspace_config=os.environ.get(CUBLAS_WORKSPACE_CONFIG_ENV, ""),
        warnings=warnings,
    )
