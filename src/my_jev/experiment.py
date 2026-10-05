from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .data_audit import audit_dataset_splits
from .locking import (
    Lease,
    LeaseSet,
    ResourceRequest,
    atomic_write_json,
)
from .manifest import dataset_manifest
from .promotion import evaluate_promotion, evaluate_regression
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
    fleet_states: str | None = None
    fleet_family_data: str | None = None
    fleet_family_batch_size: int = 8
    fleet_min_hard_failure_defer_rate: float = 1.0
    fleet_min_capacity_boundary_accuracy: float = 1.0
    fleet_min_node_permutation_agreement: float = 1.0
    fleet_min_pressure_sensitivity_rate: float = 0.0
    fleet_min_family_invariant_top1_agreement: float = 1.0
    fleet_max_family_invariant_mean_abs_probability_delta: float = 1e-5
    fleet_min_family_semantic_new_target_accuracy: float = 0.80
    fleet_min_family_semantic_probability_direction_rate: float = 0.80
    fleet_min_family_semantic_unchanged_top1_agreement: float = 0.95
    fleet_min_family_success_rate: float = 0.75


class RegressionSpec(BaseModel):
    require_same_evaluation_hashes: bool = True
    max_accuracy_drop: float | None = 0.02
    max_ece_increase: float | None = 0.02
    max_nll_increase: float | None = 0.05
    max_brier_increase: float | None = 0.05
    max_policy_consistency_violation_rate_increase: float | None = 0.01
    max_batch_ms_p95_ratio: float | None = None
    max_decisions_per_second_drop_fraction: float | None = None

    def gates(self) -> dict[str, float]:
        payload = self.model_dump()
        payload.pop(
            "require_same_evaluation_hashes"
        )
        return {
            name: float(value)
            for name, value in payload.items()
            if value is not None
        }


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
    regression: RegressionSpec = Field(
        default_factory=RegressionSpec
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


def _benchmark_input_manifests(
    spec: ExperimentSpec,
    root: Path,
) -> dict[str, dict[str, object]]:
    inputs: dict[
        str,
        dict[str, object],
    ] = {}

    for name, value in (
        (
            "fleet_states",
            spec.benchmark.fleet_states,
        ),
        (
            "fleet_family_data",
            spec.benchmark.fleet_family_data,
        ),
    ):
        if not value:
            continue
        path = _resolve(
            value,
            root,
        )
        payload = path.read_bytes()
        inputs[name] = {
            "path": str(path),
            "sha256": _sha256_bytes(
                payload
            ),
            "bytes": len(payload),
            "lines": len(
                [
                    line
                    for line in (
                        payload.decode(
                            "utf-8"
                        ).splitlines()
                    )
                    if line.strip()
                ]
            ),
        }

    return inputs


def _run_id(
    name: str,
    spec_sha256: str,
) -> str:
    timestamp = (
        dt.datetime.now(
            dt.UTC
        )
        .strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
    )
    return (
        f"{_safe_name(name)}-"
        f"{timestamp}-"
        f"{spec_sha256[:8]}"
    )


def canonical_device_id(device: str) -> str:
    """Normalise a device id to one spelling, refusing anything ambiguous.

    Lock names are derived from this, so it has to be a *function* of the card
    rather than of how the caller happened to spell it. ``_safe_name`` collapses
    ``cuda:0``, ``cuda/0``, ``cuda.0``, ``cuda-0`` and ``cuda 0`` onto one lock, so
    two spellings of the same card -- or two different cards spelled similarly --
    would contend against each other for no reason. Canonicalising first means
    distinct cards always get distinct locks.

    Rejects rather than normalises where the intent is unclear. Returns
    ``cuda:N`` for every accepted form.
    """
    text = str(device).strip()
    if text == "":
        raise ValueError("device id must be non-empty")
    if text == "cuda":
        return "cuda:0"
    if text.isdigit():
        return f"cuda:{text}"
    if text.startswith("cuda:"):
        index = text.removeprefix("cuda:").strip()
        if index.isdigit():
            return f"cuda:{index}"
    raise ValueError(
        f"unsupported device id {device!r}; expected 'cuda:N' or a bare index"
    )


def stage_device_pin(
    device: str,
    *,
    detected: Sequence[str] | None = None,
) -> str:
    """Translate a planned device id into a ``CUDA_VISIBLE_DEVICES`` value.

    Every module in this repository resolves ``torch.device("cuda")`` to whatever
    the visible set allows, and none of them reads an explicit index. Pinning
    ``CUDA_VISIBLE_DEVICES`` for the stage process is therefore the only way to
    make a device assignment true -- and it needs no change to the nine modules
    that pick a device.

    A device that is not visible to this host is refused rather than silently
    remapped: a plan claiming ``cuda:3`` on a one-GPU box should fail loudly,
    because running it anyway would put the work somewhere the plan did not say.
    """
    canonical = canonical_device_id(device)
    index = canonical.removeprefix("cuda:")
    if detected is not None and canonical not in set(detected):
        raise ValueError(
            f"planned device {canonical} is not visible here; "
            f"detected: {sorted(detected) or 'none'}"
        )
    return index


def _stage(
    name: str,
    command: list[str],
    *,
    run_dir: Path,
    dry_run: bool,
    device: str | None = None,
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
            # Recorded so a run can be reproduced or diagnosed later without
            # guessing which physical GPU it used.
            "device": device,
            "CUDA_VISIBLE_DEVICES": device,
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
    returncode: int | None = None
    # Wall-clock the stage, so gpu_plan can read real per-stage durations instead
    # of inferring them from a shape model. Recorded before the failure check:
    # a stage that died still consumed GPU, and a duration nobody wrote down is
    # the main thing that made the planner's estimates worth calibrating at all.
    # subprocess.run replaces the whole environment when `env` is given, so the
    # pin has to be layered onto the current one rather than replacing it.
    stage_env = None
    if device is not None:
        stage_env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": stage_device_pin(device),
        }

    started_at = time.time()
    try:
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
                env=stage_env,
            )
        returncode = process.returncode
    finally:
        finished_at = time.time()
        atomic_write_json(
            stages
            / f"{name}.timing.json",
            {
                "name": name,
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_seconds": round(
                    finished_at - started_at,
                    6,
                ),
                "completed": returncode is not None and returncode == 0,
            },
        )
    if returncode != 0:
        raise RuntimeError(
            f"stage {name} failed "
            f"with exit code "
            f"{returncode}; "
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


def _fleet_benchmark_command(
    spec: ExperimentSpec,
    root: Path,
    run_dir: Path,
) -> list[str] | None:
    if not spec.benchmark.fleet_states:
        return None

    command = [
        sys.executable,
        "-m",
        "my_jev.fleet_benchmark",
        "--checkpoint",
        str(
            run_dir
            / "checkpoints"
            / "best"
        ),
        "--states",
        str(
            _resolve(
                spec.benchmark.fleet_states,
                root,
            )
        ),
        "--calibration",
        str(
            run_dir
            / "calibration.json"
        ),
        "--output",
        str(
            run_dir
            / "fleet-benchmark.json"
        ),
        "--min-hard-failure-defer-rate",
        str(
            spec.benchmark
            .fleet_min_hard_failure_defer_rate
        ),
        "--min-capacity-boundary-accuracy",
        str(
            spec.benchmark
            .fleet_min_capacity_boundary_accuracy
        ),
        "--min-node-permutation-agreement",
        str(
            spec.benchmark
            .fleet_min_node_permutation_agreement
        ),
        "--min-pressure-sensitivity-rate",
        str(
            spec.benchmark
            .fleet_min_pressure_sensitivity_rate
        ),
        "--min-family-invariant-top1-agreement",
        str(
            spec.benchmark
            .fleet_min_family_invariant_top1_agreement
        ),
        (
            "--max-family-invariant-"
            "mean-abs-probability-delta"
        ),
        str(
            spec.benchmark
            .fleet_max_family_invariant_mean_abs_probability_delta
        ),
        "--min-family-semantic-new-target-accuracy",
        str(
            spec.benchmark
            .fleet_min_family_semantic_new_target_accuracy
        ),
        (
            "--min-family-semantic-"
            "probability-direction-rate"
        ),
        str(
            spec.benchmark
            .fleet_min_family_semantic_probability_direction_rate
        ),
        (
            "--min-family-semantic-"
            "unchanged-top1-agreement"
        ),
        str(
            spec.benchmark
            .fleet_min_family_semantic_unchanged_top1_agreement
        ),
        "--min-family-success-rate",
        str(
            spec.benchmark
            .fleet_min_family_success_rate
        ),
    ]

    if (
        spec.benchmark
        .fleet_family_data
    ):
        command.extend(
            [
                "--family-data",
                str(
                    _resolve(
                        spec.benchmark
                        .fleet_family_data,
                        root,
                    )
                ),
                "--family-batch-size",
                str(
                    spec.benchmark
                    .fleet_family_batch_size
                ),
            ]
        )

    return command


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


def _register_run(
    *,
    registry_path: Path,
    lock_dir: Path,
    entry: ExperimentEntry,
    spec: ExperimentSpec,
) -> str | None:
    with Lease(
        lock_dir,
        ResourceRequest(
            "registry"
        ),
        command="register experiment",
    ):
        registry = ExperimentRegistry(
            registry_path
        )
        parent_run_id = (
            _resolve_parent(
                registry,
                spec,
            )
        )
        entry.parent_run_id = (
            parent_run_id
        )
        registry.register(
            entry
        )
    return parent_run_id


def _update_run(
    *,
    registry_path: Path,
    lock_dir: Path,
    run_id: str,
    **changes: object,
) -> None:
    with Lease(
        lock_dir,
        ResourceRequest(
            "registry"
        ),
        command=(
            f"update experiment {run_id}"
        ),
    ):
        registry = ExperimentRegistry(
            registry_path
        )
        registry.update(
            run_id,
            **changes,
        )


def _parent_regression(
    *,
    registry_path: Path,
    parent_run_id: str | None,
    candidate_entry: ExperimentEntry,
    candidate_benchmark: dict[str, object],
    regression: RegressionSpec,
) -> dict[str, object]:
    if parent_run_id is None:
        return {
            "passed": True,
            "status": "no_parent",
            "failed": [],
            "gates": [],
        }

    registry = ExperimentRegistry(
        registry_path
    )
    parent = registry.get(
        parent_run_id
    )
    if parent is None:
        return {
            "passed": False,
            "status": "missing_parent",
            "parent_run_id": parent_run_id,
            "failed": [
                "parent_run_missing"
            ],
            "gates": [],
        }

    same_test = (
        parent.test_sha256
        == candidate_entry.test_sha256
    )
    same_calibration = (
        parent.calibration_sha256
        == candidate_entry.calibration_sha256
    )
    comparable = (
        same_test
        and same_calibration
    )
    if (
        regression
        .require_same_evaluation_hashes
        and not comparable
    ):
        return {
            "passed": False,
            "status": "evaluation_hash_mismatch",
            "parent_run_id": (
                parent_run_id
            ),
            "same_test_split": same_test,
            "same_calibration_split": (
                same_calibration
            ),
            "failed": [
                "evaluation_hash_mismatch"
            ],
            "gates": [],
        }

    if (
        not parent.benchmark_path
        or not Path(
            parent.benchmark_path
        ).exists()
    ):
        return {
            "passed": False,
            "status": "missing_parent_benchmark",
            "parent_run_id": (
                parent_run_id
            ),
            "failed": [
                "parent_benchmark_missing"
            ],
            "gates": [],
        }

    baseline = json.loads(
        Path(
            parent.benchmark_path
        ).read_text(
            encoding="utf-8"
        )
    )
    result = evaluate_regression(
        candidate_benchmark,
        baseline,
        regression.gates(),
    )
    result.update(
        {
            "status": "evaluated",
            "parent_run_id": (
                parent_run_id
            ),
            "same_test_split": same_test,
            "same_calibration_split": (
                same_calibration
            ),
        }
    )
    return result


def device_assignments_from_plan(
    plan_path: str | Path,
) -> dict[str, str]:
    """Read per-stage device assignments out of a ``my-jev-gpu-plan`` document.

    This is what makes the plan executable rather than advisory in name only. The
    planner says which device each stage belongs on; this carries that into the
    environment the stage actually runs in, so the assignment is enforced rather
    than documented.

    Stages the plan did not place are omitted rather than defaulted: an unplanned
    stage still runs, unpinned, and its ``command.json`` records that no device
    was assigned.
    """
    from .gpu_plan import ModelVersionPlan

    plan = ModelVersionPlan.model_validate_json(
        Path(plan_path).read_text(encoding="utf-8")
    )
    assignments: dict[str, str] = {}
    for stage in plan.stages:
        if stage.gpu_ids:
            # First device is the primary; the placement already ordered them by
            # descending free time, so this is the one the plan meant.
            assignments[stage.kind.value] = stage.gpu_ids[0]
    return assignments


def gpu_lease_requests(device: str | None) -> list[ResourceRequest]:
    """Leases that reserve a GPU for this run.

    With a known device, a run takes a *shared* claim on the pool plus an
    exclusive claim on that one card. With no device it falls back to the
    historical exclusive claim on the pool.

    The pairing is what makes the change safe to deploy while older runs are still
    in flight. ``flock`` excludes exclusive against shared, so:

    * an old run's exclusive ``gpu`` blocks every new run -- the conservative
      outcome, and exactly what happens today;
    * two new runs on different cards hold ``gpu`` concurrently and conflict only
      on their own card.

    So nothing has to be decided in advance and nothing has to be drained. As old
    runs finish, concurrency appears on its own; while any remain, behaviour is
    unchanged. A new scheme that merely added ``gpu:0`` would not have had that
    property: ``gpu`` and ``gpu:0`` do not conflict, so an old run and a new run
    could land on one card.
    """
    if device is None:
        return [ResourceRequest("gpu")]
    # Canonicalised before it becomes a lock name, and refused here rather than
    # downstream: by this point the spelling is load-bearing, and a typo would
    # otherwise cost concurrency silently rather than loudly.
    canonical = canonical_device_id(device)
    return [
        # Shared: "somebody is in the pool", compatible with other new runs.
        ResourceRequest("gpu", shared=True),
        # Exclusive: this specific card, so two runs cannot share it.
        ResourceRequest(f"gpu:{canonical}", shared=False),
    ]


def run_experiment(
    spec_path: str | Path,
    *,
    dry_run: bool = False,
    blocking: bool = False,
    device_assignments: Mapping[str, str] | None = None,
    gpu_plan: str | Path | None = None,
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
    dataset_paths = {
        name: _resolve(
            getattr(
                spec.data,
                name,
            ),
            root,
        )
        for name in (
            "train",
            "validation",
            "calibration",
            "test",
        )
    }
    data_audit = audit_dataset_splits(
        dataset_paths
    )
    manifests = _dataset_manifests(
        spec,
        root,
    )
    benchmark_inputs = (
        _benchmark_input_manifests(
            spec,
            root,
        )
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
    registry_path = _resolve(
        spec.experiment.registry_path,
        root,
    )
    lock_dir = _resolve(
        spec.experiment.lock_dir,
        root,
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
        parent_run_id=None,
    )
    parent_run_id = _register_run(
        registry_path=registry_path,
        lock_dir=lock_dir,
        entry=entry,
        spec=spec,
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
        "benchmark_inputs": (
            benchmark_inputs
        ),
        "data_audit": data_audit,
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
    fleet_command = _fleet_benchmark_command(spec, root, run_dir)
    if fleet_command is not None:
        commands["fleet_benchmark"] = fleet_command

    # A plan supplies the device per stage. An explicit mapping wins, so a caller
    # can override a plan without editing it.
    devices: dict[str, str] = {}
    if gpu_plan is not None:
        devices.update(device_assignments_from_plan(gpu_plan))
    if device_assignments:
        devices.update({str(k): str(v) for k, v in device_assignments.items()})
    unknown = sorted(set(devices) - set(commands))
    if unknown:
        # A plan naming a stage this pipeline never runs is a stale plan, and
        # silently ignoring it would leave the operator believing it applied.
        raise ValueError(
            f"device assignment names stages this pipeline does not run: {unknown}"
        )

    if dry_run:
        for name, command in (
            commands.items()
        ):
            _stage(
                name,
                command,
                run_dir=run_dir,
                dry_run=True,
                # Recorded even though nothing runs, so a dry run shows the
                # device each stage would have used.
                device=devices.get(name),
            )
        _update_run(
            registry_path=registry_path,
            lock_dir=lock_dir,
            run_id=run_id,
            status="dry_run",
        )
        return {
            "run_id": run_id,
            "run_dir": str(
                run_dir
            ),
            "dry_run": True,
            "data_audit": data_audit,
            "commands": commands,
        }

    leases = LeaseSet(
        lock_dir,
        [
            ResourceRequest(
                "datasets",
                shared=True,
            ),
            *gpu_lease_requests(devices.get("train")),
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
                    device=devices.get(name),
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
        absolute = (
            evaluate_promotion(
                benchmark,
                spec.gates,
            )
        )
        regression = _parent_regression(
            registry_path=registry_path,
            parent_run_id=parent_run_id,
            candidate_entry=entry,
            candidate_benchmark=benchmark,
            regression=spec.regression,
        )
        fleet_promotion = {"passed": True, "status": "not_configured", "failed": []}
        fleet_path = run_dir / "fleet-benchmark.json"
        if fleet_command is not None:
            fleet_artifact = json.loads(fleet_path.read_text(encoding="utf-8"))
            fleet_promotion = fleet_artifact["promotion"]
        promotion = {
            "passed": (
                bool(absolute["passed"])
                and bool(regression["passed"])
                and bool(fleet_promotion["passed"])
            ),
            "absolute": absolute,
            "regression": regression,
            "fleet": fleet_promotion,
            "failed": [
                *[
                    f"absolute:{name}"
                    for name in absolute[
                        "failed"
                    ]
                ],
                *[
                    f"regression:{name}"
                    for name in regression["failed"]
                ],
                *[
                    f"fleet:{name}"
                    for name in fleet_promotion.get("failed", [])
                ],
            ],
            "run_id": run_id,
            "evaluated_at": (
                dt.datetime.now(
                    dt.UTC
                ).isoformat()
            ),
        }
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
        _update_run(
            registry_path=registry_path,
            lock_dir=lock_dir,
            run_id=run_id,
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
        _update_run(
            registry_path=registry_path,
            lock_dir=lock_dir,
            run_id=run_id,
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
        "--gpu-plan",
        help=(
            "A my-jev-gpu-plan document. Each stage is pinned to the device the "
            "plan assigned via CUDA_VISIBLE_DEVICES, so the assignment is "
            "enforced rather than documented."
        ),
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
        gpu_plan=args.gpu_plan,
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
