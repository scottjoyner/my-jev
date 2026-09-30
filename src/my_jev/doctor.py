from __future__ import annotations

import argparse
import importlib.metadata
import json
from typing import Any

import torch


def _distribution_version(
    name: str,
) -> str | None:
    try:
        return (
            importlib.metadata.version(
                name
            )
        )
    except (
        importlib.metadata
        .PackageNotFoundError
    ):
        return None


def hardware_report() -> dict[str, Any]:
    cuda_available = (
        torch.cuda.is_available()
    )
    devices: list[
        dict[str, object]
    ] = []
    if cuda_available:
        for index in range(
            torch.cuda.device_count()
        ):
            properties = (
                torch.cuda
                .get_device_properties(
                    index
                )
            )
            devices.append(
                {
                    "index": index,
                    "name": (
                        properties.name
                    ),
                    "total_memory_bytes": (
                        int(
                            properties
                            .total_memory
                        )
                    ),
                    "total_memory_gib": (
                        float(
                            properties
                            .total_memory
                        )
                        / (
                            1024
                            ** 3
                        )
                    ),
                    "major": int(
                        properties.major
                    ),
                    "minor": int(
                        properties.minor
                    ),
                }
            )

    bf16_supported = bool(
        cuda_available
        and torch.cuda
        .is_bf16_supported()
    )
    return {
        "torch_version": (
            torch.__version__
        ),
        "hip_version": getattr(
            torch.version,
            "hip",
            None,
        ),
        "cuda_runtime_version": (
            getattr(
                torch.version,
                "cuda",
                None,
            )
        ),
        "accelerator_available": (
            cuda_available
        ),
        "device_count": len(
            devices
        ),
        "devices": devices,
        "bf16_supported": (
            bf16_supported
        ),
        "packages": {
            "transformers": (
                _distribution_version(
                    "transformers"
                )
            ),
            "peft": (
                _distribution_version(
                    "peft"
                )
            ),
            "accelerate": (
                _distribution_version(
                    "accelerate"
                )
            ),
        },
    }


def validate_environment(
    report: dict[str, Any],
    *,
    require_gpu: bool = False,
    require_bf16: bool = False,
    require_causal: bool = False,
) -> list[str]:
    issues: list[str] = []
    if (
        require_gpu
        and not report[
            "accelerator_available"
        ]
    ):
        issues.append(
            "no PyTorch accelerator "
            "is available"
        )
    if (
        require_bf16
        and not report[
            "bf16_supported"
        ]
    ):
        issues.append(
            "BF16 is not reported "
            "as supported"
        )
    if require_causal:
        packages = report[
            "packages"
        ]
        assert isinstance(
            packages,
            dict,
        )
        for name in (
            "transformers",
            "peft",
            "accelerate",
        ):
            if not packages.get(
                name
            ):
                issues.append(
                    "missing causal-lane "
                    f"dependency: {name}"
                )
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report my-jev training "
            "hardware/runtime readiness"
        )
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
    )
    parser.add_argument(
        "--require-bf16",
        action="store_true",
    )
    parser.add_argument(
        "--require-causal",
        action="store_true",
    )
    args = parser.parse_args()

    report = hardware_report()
    issues = validate_environment(
        report,
        require_gpu=args.require_gpu,
        require_bf16=(
            args.require_bf16
        ),
        require_causal=(
            args.require_causal
        ),
    )
    payload = {
        **report,
        "ready": not issues,
        "issues": issues,
    }
    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
    )
    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
