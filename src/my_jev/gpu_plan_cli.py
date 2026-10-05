"""Plan GPU time for a model version from a fine-tuning job.

Advisory. Prints a plan and acquires nothing; see ``gpu_plan.PlanAuthority``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .experiment import load_experiment_spec
from .gpu_inventory import default_lock_dir, inventory_from_file, local_gpu_inventory
from .gpu_plan import (
    GpuCandidate,
    ReadinessVerdict,
    StageKind,
    build_model_version_plan,
    calibration_from_registry,
    readiness_from_run,
)


def _inventory(
    path: Path | None,
    *,
    spec: object,
    args: argparse.Namespace,
) -> list[GpuCandidate]:
    """Read the GPU inventory, from a document or by probing this host.

    Deliberately separate from the budget. The planner allocates only from what
    it is told is free, so a stale inventory under-allocates rather than
    double-booking a device somebody is using.

    With ``--inventory local`` the devices come from PyTorch or ``nvidia-smi``
    and the busy-or-free answer comes from the same ``gpu`` lease
    ``experiment.py`` takes before it runs anything, so the probe cannot disagree
    with the run it is planning for.
    """
    if path is not None and str(path) != "local":
        return inventory_from_file(path)
    if path is None and not args.probe_local_inventory:
        return []
    registry = getattr(getattr(spec, "experiment", None), "registry_path", None)
    # An explicit --lock-dir wins. The previous form tested whether *a* lock dir
    # applied but then always passed the derived one, so --lock-dir was silently
    # ignored and the probe answered about a directory the run would not use.
    lock_dir = args.lock_dir or (default_lock_dir(registry) if registry else None)
    return local_gpu_inventory(
        lock_dir=lock_dir,
        horizon_gpu_minutes=args.horizon_gpu_minutes,
        horizon_is_per_device=args.horizon_is_per_device,
        untrusted=tuple(filter(None, (args.untrusted_gpu or "").split(","))),
    )


def _measured(path: Path | None) -> dict[StageKind, float]:
    """Read measured minutes from completed runs, keyed by stage.

    The escape hatch from the estimate. A plan rebuilt after stages have run
    budgets what they actually cost, so repeated planning does not inflate the
    total with work that was never justified twice.
    """
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        StageKind(key): float(value)
        for key, value in payload.items()
        if key in {stage.value for stage in StageKind}
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan GPU time for the model version a fine-tuning job would produce. "
            "Prints a plan; acquires no GPU and grants no authority."
        )
    )
    parser.add_argument("spec", type=Path, help="Experiment spec TOML.")
    parser.add_argument(
        "--budget-gpu-minutes",
        type=float,
        required=True,
        help=(
            "GPU-minutes available for this version. The train/benchmark/evaluate "
            "chain is all-or-nothing: a budget that funds part of it funds none."
        ),
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        help=(
            "JSON GPU inventory: {gpus: [{gpu_id, free_gpu_minutes, trusted}]}. "
            "Pass 'local' to probe this host instead of asserting an inventory."
        ),
    )
    parser.add_argument(
        "--probe-local-inventory",
        action="store_true",
        help="Detect devices on this host and probe the GPU lease.",
    )
    parser.add_argument(
        "--lock-dir",
        help=(
            "Lease directory to probe for GPU availability. Defaults to the "
            "lock directory implied by the spec's registry_path."
        ),
    )
    parser.add_argument(
        "--horizon-gpu-minutes",
        type=float,
        default=0.0,
        help=(
            "How long a device may be assumed free. Zero is the safe default: "
            "the plan schedules nothing rather than inventing availability."
        ),
    )
    parser.add_argument(
        "--horizon-is-per-device",
        action="store_true",
        help=(
            "Treat --horizon-gpu-minutes as per device rather than dividing it "
            "across them. The GPU lease is a single exclusive resource, so the "
            "total is the honest quantity by default."
        ),
    )
    parser.add_argument(
        "--untrusted-gpu",
        help="Comma-separated device ids never to allocate, e.g. 'cuda:1'.",
    )
    parser.add_argument(
        "--measured",
        type=Path,
        help="JSON {stage: minutes} from completed runs; overrides the estimate.",
    )
    parser.add_argument(
        "--calibration-registry",
        type=Path,
        help=(
            "Experiment registry to calibrate stage costs against, using runs "
            "that completed the chain. Per-stage figures stay inferred: the run "
            "lifecycle records no per-stage timestamps."
        ),
    )
    parser.add_argument(
        "--calibration-run-dir",
        help=(
            "Run directory to read real per-stage timings from. Without this, "
            "calibration falls back to the chain total and marks the per-stage "
            "split as inferred."
        ),
    )
    parser.add_argument(
        "--readiness-run-dir",
        help=(
            "Run directory to read scale readiness from. When given and readiness "
            "does not pass, no training GPU is planned."
        ),
    )
    parser.add_argument(
        "--readiness-passed",
        action="store_true",
        help=(
            "Assert readiness passed without reading a run. Prefer "
            "--readiness-run-dir: an assertion cannot be checked."
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec, _sha = load_experiment_spec(args.spec)

    readiness: ReadinessVerdict | None = None
    if args.readiness_run_dir:
        readiness = readiness_from_run(args.readiness_run_dir)
    elif args.readiness_passed:
        readiness = ReadinessVerdict(
            completed=True,
            passed=True,
            source="operator-assertion",
        )

    payload = spec.model_dump(mode="json")
    calibration = (
        calibration_from_registry(
            args.calibration_registry,
            planned_job=payload,
            run_dir=args.calibration_run_dir,
        )
        if args.calibration_registry is not None
        else None
    )

    plan = build_model_version_plan(
        payload,
        pool=_inventory(args.inventory, spec=spec, args=args),
        budget_gpu_minutes=args.budget_gpu_minutes,
        readiness=readiness,
        measured=_measured(args.measured),
        calibration=calibration,
    )

    encoded = json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)

    # Non-zero unless the plan is both affordable and placeable, so a shell can
    # gate on it without parsing the document. Budget alone is not enough: a
    # plan that fits its budget with every GPU busy cannot be run. The plan is
    # still printed either way, because "why is this not actionable" is the
    # useful output in the failing case.
    return 0 if (plan.within_budget and plan.fully_placed) else 2


if __name__ == "__main__":
    sys.exit(main())
