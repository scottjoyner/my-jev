from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


RECEIPT_VERSION = "r9700-baseline-evidence-v1"
REQUIRED_FILES = (
    "manifest.json",
    "checkpoints/best/model.pt",
    "checkpoints/best/my_jev_config.json",
    "calibration.json",
    "benchmark.json",
    "promotion.json",
    "r9700-doctor.json",
    "pip-freeze.txt",
    "r9700-baseline-receipt.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_baseline_run(
    run_dir: str | Path,
    *,
    expected_sha: str | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    if not root.is_dir():
        raise ValueError(f"run directory does not exist: {root}")

    missing = [
        relative
        for relative in REQUIRED_FILES
        if not (root / relative).is_file()
    ]
    if missing:
        raise ValueError(
            "baseline run is incomplete; missing: " + ", ".join(missing)
        )

    manifest = _load_json(root / "manifest.json")
    config = _load_json(
        root / "checkpoints/best/my_jev_config.json"
    )
    benchmark = _load_json(root / "benchmark.json")
    promotion = _load_json(root / "promotion.json")
    doctor = _load_json(root / "r9700-doctor.json")
    receipt = _load_json(root / "r9700-baseline-receipt.json")

    git_state = manifest.get("git")
    if not isinstance(git_state, dict):
        git_state = manifest.get("git_state")
    if not isinstance(git_state, dict):
        raise ValueError("manifest is missing git provenance")

    revision = str(
        git_state.get("revision")
        or git_state.get("sha")
        or git_state.get("commit")
        or ""
    )
    dirty = bool(git_state.get("dirty", False))
    if not revision:
        raise ValueError("manifest git provenance has no revision")
    if dirty:
        raise ValueError("baseline manifest records a dirty checkout")
    if expected_sha is not None and revision != expected_sha:
        raise ValueError(
            f"baseline SHA mismatch: expected {expected_sha}, got {revision}"
        )
    if str(receipt.get("git_sha", "")) != revision:
        raise ValueError("receipt git_sha does not match manifest revision")

    extra = config.get("extra")
    if not isinstance(extra, dict):
        raise ValueError("checkpoint config is missing effective run config")
    expected_config = {
        "device": "cuda",
        "epochs": 1,
        "batch_size": 2,
        "grad_accum": 8,
        "max_state_length": 2048,
        "max_candidate_length": 192,
        "bf16_requested": True,
        "bf16_enabled": True,
        "freeze_backbone": False,
        "gradient_checkpointing": True,
    }
    mismatches = {
        key: {"expected": value, "actual": extra.get(key)}
        for key, value in expected_config.items()
        if extra.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "checkpoint effective configuration mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )

    hip_version = doctor.get("hip_version")
    if not hip_version:
        raise ValueError("R9700 doctor evidence has no HIP version")

    devices = doctor.get("devices")
    if not isinstance(devices, list):
        devices = doctor.get("cuda_devices")
    if not isinstance(devices, list):
        devices = []
    r9700 = [
        item
        for item in devices
        if isinstance(item, dict)
        and "r9700" in str(
            item.get("name") or item.get("device_name") or ""
        ).lower()
    ]
    if not r9700:
        raise ValueError("doctor evidence does not identify an R9700 device")

    artifact_hashes: dict[str, dict[str, Any]] = {}
    for relative in REQUIRED_FILES:
        path = root / relative
        artifact_hashes[relative] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }

    receipt_artifacts = receipt.get("artifacts")
    if isinstance(receipt_artifacts, dict):
        for relative, evidence in receipt_artifacts.items():
            if relative not in artifact_hashes or not isinstance(evidence, dict):
                continue
            recorded = str(evidence.get("sha256", ""))
            if recorded and recorded != artifact_hashes[relative]["sha256"]:
                raise ValueError(
                    f"receipt artifact hash mismatch for {relative}"
                )

    promotion_status = str(
        promotion.get("status")
        or promotion.get("decision")
        or "unknown"
    )
    report = {
        "version": RECEIPT_VERSION,
        "evidence_only": True,
        "runtime_authority_changed": False,
        "run_dir": str(root),
        "git_sha": revision,
        "checkpoint": str(root / "checkpoints/best"),
        "calibration": str(root / "calibration.json"),
        "benchmark": str(root / "benchmark.json"),
        "promotion_status": promotion_status,
        "benchmark_metrics": benchmark.get("metrics", benchmark),
        "hip_version": hip_version,
        "r9700_devices": r9700,
        "artifacts": artifact_hashes,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a completed exact-SHA R9700 baseline artifact bundle "
            "without changing Hermes/AssistX runtime authority"
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-sha")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = validate_baseline_run(
        args.run_dir,
        expected_sha=args.expected_sha,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
