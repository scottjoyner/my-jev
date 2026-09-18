from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .locking import (
    LeaseSet,
    ResourceRequest,
    atomic_write_json,
)
from .manifest import dataset_manifest
from .promotion import evaluate_promotion
from .registry import (
    ExperimentEntry,
    ExperimentRegistry,
)


class ExperimentMeta(BaseModel):
    name: str
    output_root: str = "runs/experiments"
    registry_path: str = "runs/experiments/registry.json"
    lock_dir: str = "runs/experiments/.locks"
    parent: str | None = "latest_promoted"


class ModelSpec(BaseModel):
    backend: Literal[
        "encoder_option_query",
        "causal_scalar",
    ] = "encoder_option_query"
    backbone: str = (
        "answerdotai/ModernBERT-base"
    )
    head_kind: Literal[
        "option_query",
        "legacy",
    ] = "option_query"
    head_rank: int | None = 256
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"


class DataSpec(BaseModel):
    train: str
    validation: str
    calibration: str
    test: str


class TrainSpec(BaseModel):
    epochs: int = 3
    batch_size: int = 2
    grad_accum: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_state_length: int = 2048
    max_candidate_length: int = 192
    max_sequence_length: int = 1024
    seed: int = 17
    bf16: bool = True
    freeze_backbone: bool = False
    gradient_checkpointing: bool = True


class BenchmarkSpec(BaseModel):
    batch_size: int = 8


class ExperimentSpec(BaseModel):
    experiment: ExperimentMeta
    model: ModelSpec = Field(
        default_factory=ModelSpec
    )
    data: DataSpec
    train: TrainSpec = Field(
        default_factory=TrainSpec
    )
    benchmark: BenchmarkSpec = Field(
        default_factory=BenchmarkSpec
    )
    gates: dict[
        str,
        float,
    ] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_gates(
        self,
    ) -> ExperimentSpec:
        if not self.gates:
            raise ValueError(
                "experiment must define "
                "explicit promotion gates"
            )
        return self


def _sha256_bytes(
    payload: bytes,
) -> str:
    return hashlib.sha256(
        payload
    ).hexdigest()


def _safe_name(
    value: str,
) -> str:
    cleaned = "".join(
        character
        if character.isalnum()
        or character in {"-", "_"}
        else "-"
        for character in value
    ).strip("-")
    return cleaned or "experiment"


def load_experiment_spec(
    path: str | Path,
) -> tuple[
    ExperimentSpec,
    str,
]:
    source = Path(path)
    raw = source.read_bytes()
    payload = tomllib.loads(
        raw.decode("utf-8")
    )
    return (
        ExperimentSpec.model_validate(
            payload
        ),
        _sha256_bytes(raw),
    )


def _repo_root() -> Path:
    return Path.cwd().resolve()


def _resolve(
    value: str,
    root: Path,
) -> Path:
    path = Path(value)
    return (
        path.resolve()
        if path.is_absolute()
        else (root / path).resolve()
    )


def _git_state(
    root: Path,
) -> dict[str, object]:
    def run(
        *args: str,
    ) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    status = run(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    return {
        "revision": run(
            "rev-parse",
            "HEAD",
        ),
        "branch": run(
            "branch",
            "--show-current",
        ),
        "dirty": (
            bool(status)
            if status is not None
            else None
        ),
        "status": status,
    }


def _dataset_manifests(
    spec: ExperimentSpec,
    root: Path,
) -> dict[
    str,
    dict[str, object],
]:
    manifests = {}
    for name in (
        "train",
        "validation",
        "calibration",
        "test",
    ):
        path = _resolve(
            getattr(
                spec.data,
                name,
            ),
            root,
        )
        manifests[
            name
        ] = dataset_manifest(
            path
        )
    return manifests


def _run_id(
    name: str,
    spec_sha256: str,
) -> str:
    timestamp = (
        dt.datetime.now(
            dt.UTC
        )
        .strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )
    return (
        f"{_safe_name(name)}-"
        f"{timestamp}-"
        f"{spec_sha256[:8]}"
    )


def _stage(
    name: str,
    command: list[str],
    *,
    run_dir: Path,
    dry_run: bool,
) -> None:
    stages = run_dir / "stages"
    stages.mkdir(
        parents=True,
        exist_ok=True,
    )
    atomic_write_json(
        stages
        / f"{name}.command.json",
        {
            "name": name,
            "argv": command,
            "cwd": os.getcwd(),
        },
    )
    if dry_run:
        print(
            "$ "
            + " ".join(command)
        )
        return

    log_path = (
        stages
        / f"{name}.log"
    )
    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"stage {name} failed "
            f"with exit code "
            f"{process.returncode}; "
            f"see {log_path}"
        )


def _train_command(
    spec: ExperimentSpec,
    root: Path,
    run_dir: Path,
) -> list[str]:
    common = [
        "--train",
        str(
            _resolve(
                spec.data.train,
                root,
            )
        ),
        "--valid",
        str(
            _resolve(
                spec.data.validation,
                root,
            )
        ),
        "--output",
        str(
            run_dir
            / "checkpoints"
        ),
        "--backbone",
        spec.model.backbone,
        "--epochs",
        str(
            spec.train.epochs
        ),
        "--batch-size",
        str(
            spec.train.batch_size
        ),
        "--grad-accum",
        str(
            spec.train.grad_accum
        ),
        "--lr",
        str(
            spec.train.learning_rate
        ),
        "--weight-decay",
        str(
            spec.train.weight_decay
        ),
        "--seed",
        str(
            spec.train.seed
        ),
    ]

    if (
        spec.model.backend
        == "causal_scalar"
    ):
        command = [
            sys.executable,
            "-m",
            "my_jev.train_causal",
            *common,
            "--max-length",
            str(
                spec.train
                .max_sequence_length
            ),
            "--lora-r",
            str(
                spec.model.lora_r
            ),
            "--lora-alpha",
            str(
                spec.model.lora_alpha
            ),
            "--lora-dropout",
            str(
                spec.model
                .lora_dropout
            ),
            "--target-modules",
            spec.model.target_modules,
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "my_jev.train",
            *common,
            "--head-kind",
            spec.model.head_kind,
            "--max-state-length",
            str(
                spec.train
                .max_state_length
            ),
            "--max-candidate-length",
            str(
                spec.train
                .max_candidate_length
            ),
        ]
        if (
            spec.model.head_rank
            is not None
        ):
            command.extend(
                [
                    "--head-rank",
                    str(
                        spec.model
                        .head_rank
                    ),
                ]
            )
        if (
            spec.train
            .freeze_backbone
        ):
            command.append(
                "--freeze-backbone"
            )

    if spec.train.bf16:
        command.append(
            "--bf16"
        )
    if (
        spec.train
        .gradient_checkpointing
    ):
        command.append(
            "--gradient-checkpointing"
        )
    return command


def _calibration_command(
    spec: ExperimentSpec,
    root: Path,
    run_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "my_jev.calibrate",
        "--checkpoint",
        str(
            run_dir
            / "checkpoints"
            / "best"
        ),
        "--data",
        str(
            _resolve(
                spec.data.calibration,
                root,
            )
        ),
        "--output",
        str(
            run_dir
            / "calibration.json"
        ),
        "--batch-size",
        str(
            max(
                1,
                spec.benchmark
                .batch_size,
            )
        ),
    ]


def _benchmark_command(
    spec: ExperimentSpec,
    root: Path,
    run_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "my_jev.benchmark",
        "--checkpoint",
        str(
            run_dir
            / "checkpoints"
            / "best"
        ),
        "--data",
        str(
            _resolve(
                spec.data.test,
                root,
            )
        ),
        "--calibration",
        str(
            run_dir
            / "calibration.json"
        ),
        "--batch-size",
        str(
            spec.benchmark
            .batch_size
        ),
        "--output",
        str(
            run_dir
            / "benchmark.json"
        ),
    ]


def _resolve_parent(
    registry: ExperimentRegistry,
    spec: ExperimentSpec,
) -> str | None:
    parent = spec.experiment.parent
    if not parent:
        return None
    if parent == "latest_promoted":
        previous = registry.latest(
            spec.experiment.name,
            promoted_only=True,
        )
        return (
            previous.run_id
            if previous
            else None
        )
    return parent


def run_experiment(
    spec_path: str | Path,
    *,
    dry_run: bool = False,
    blocking: bool = False,
) -> dict[str, object]:
    root = _repo_root()
    spec_path = _resolve(
        str(spec_path),
        root,
    )
    spec, spec_sha256 = (
        load_experiment_spec(
            spec_path
        )
    )
    manifests = _dataset_manifests(
        spec,
        root,
    )
    run_id = _run_id(
        spec.experiment.name,
        spec_sha256,
    )
    output_root = _resolve(
        spec.experiment.output_root,
        root,
    )
    run_dir = (
        output_root
        / run_id
    )
    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )
    registry = ExperimentRegistry(
        _resolve(
            spec.experiment.registry_path,
            root,
        )
    )
    parent_run_id = (
        _resolve_parent(
            registry,
            spec,
        )
    )

    entry = ExperimentEntry(
        run_id=run_id,
        experiment=(
            spec.experiment.name
        ),
        created_at=(
            dt.datetime.now(
                dt.UTC
            ).timestamp()
        ),
        updated_at=(
            dt.datetime.now(
                dt.UTC
            ).timestamp()
        ),
        status=(
            "dry_run"
            if dry_run
            else "running"
        ),
        spec_path=str(
            spec_path
        ),
        spec_sha256=(
            spec_sha256
        ),
        model_backend=(
            spec.model.backend
        ),
        backbone=(
            spec.model.backbone
        ),
        train_sha256=str(
            manifests[
                "train"
            ]["sha256"]
        ),
        validation_sha256=str(
            manifests[
                "validation"
            ]["sha256"]
        ),
        calibration_sha256=str(
            manifests[
                "calibration"
            ]["sha256"]
        ),
        test_sha256=str(
            manifests[
                "test"
            ]["sha256"]
        ),
        run_dir=str(
            run_dir
        ),
        parent_run_id=(
            parent_run_id
        ),
    )
    registry.register(
        entry
    )

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": (
            dt.datetime.now(
                dt.UTC
            ).isoformat()
        ),
        "spec_path": str(
            spec_path
        ),
        "spec_sha256": (
            spec_sha256
        ),
        "spec": spec.model_dump(
            mode="json"
        ),
        "datasets": manifests,
        "git": _git_state(
            root
        ),
        "python": sys.version,
        "platform": sys.platform,
        "parent_run_id": (
            parent_run_id
        ),
    }
    atomic_write_json(
        run_dir
        / "manifest.json",
        manifest,
    )

    commands = {
        "train": _train_command(
            spec,
            root,
            run_dir,
        ),
        "calibrate": (
            _calibration_command(
                spec,
                root,
                run_dir,
            )
        ),
        "benchmark": (
            _benchmark_command(
                spec,
                root,
                run_dir,
            )
        ),
    }

    if dry_run:
        for name, command in (
            commands.items()
        ):
            _stage(
                name,
                command,
                run_dir=run_dir,
                dry_run=True,
            )
        registry.update(
            run_id,
            status="dry_run",
        )
        return {
            "run_id": run_id,
            "run_dir": str(
                run_dir
            ),
            "dry_run": True,
            "commands": commands,
        }

    leases = LeaseSet(
        _resolve(
            spec.experiment.lock_dir,
            root,
        ),
        [
            ResourceRequest(
                "datasets",
                shared=True,
            ),
            ResourceRequest(
                "gpu",
            ),
            ResourceRequest(
                f"run-{run_id}",
            ),
        ],
        command=(
            f"my-jev-experiment "
            f"{spec_path}"
        ),
    )

    try:
        leases.acquire(
            blocking=blocking
        )
        try:
            for name, command in (
                commands.items()
            ):
                _stage(
                    name,
                    command,
                    run_dir=run_dir,
                    dry_run=False,
                )
        finally:
            leases.release()

        benchmark_path = (
            run_dir
            / "benchmark.json"
        )
        benchmark = json.loads(
            benchmark_path.read_text(
                encoding="utf-8"
            )
        )
        promotion = (
            evaluate_promotion(
                benchmark,
                spec.gates,
            )
        )
        promotion[
            "run_id"
        ] = run_id
        promotion[
            "evaluated_at"
        ] = (
            dt.datetime.now(
                dt.UTC
            ).isoformat()
        )
        promotion_path = (
            run_dir
            / "promotion.json"
        )
        atomic_write_json(
            promotion_path,
            promotion,
        )
        passed = bool(
            promotion["passed"]
        )
        registry.update(
            run_id,
            status=(
                "candidate"
                if passed
                else "rejected"
            ),
            checkpoint_path=str(
                run_dir
                / "checkpoints"
                / "best"
            ),
            calibration_path=str(
                run_dir
                / "calibration.json"
            ),
            benchmark_path=str(
                benchmark_path
            ),
            promotion_path=str(
                promotion_path
            ),
            promoted=passed,
        )
        return {
            "run_id": run_id,
            "run_dir": str(
                run_dir
            ),
            "promotion": promotion,
        }
    except Exception as exc:
        registry.update(
            run_id,
            status="failed",
            notes=[
                f"{type(exc).__name__}: "
                f"{exc}"
            ],
        )
        atomic_write_json(
            run_dir
            / "failure.json",
            {
                "run_id": run_id,
                "error_type": (
                    type(exc).__name__
                ),
                "error": str(exc),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible my-jev "
            "train/calibrate/benchmark/"
            "promotion experiment"
        )
    )
    parser.add_argument(
        "spec",
        help="TOML experiment spec",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument(
        "--blocking",
        action="store_true",
        help=(
            "wait for resource leases "
            "instead of failing fast"
        ),
    )
    args = parser.parse_args()

    result = run_experiment(
        args.spec,
        dry_run=args.dry_run,
        blocking=args.blocking,
    )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
