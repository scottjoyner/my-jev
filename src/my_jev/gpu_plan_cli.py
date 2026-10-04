"""Plan GPU time for a model version from a fine-tuning job.

Advisory. Prints a plan and acquires nothing; see ``gpu_plan.PlanAuthority``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .experiment import load_experiment_spec
from .gpu_plan import (
    GpuCandidate,
    ReadinessVerdict,
    StageKind,
    build_model_version_plan,
    readiness_from_run,
)


def _inventory(path: Path | None) -> list[GpuCandidate]:
    """Read a caller-supplied GPU inventory.

    Deliberately a separate input from the budget. The planner allocates only
    from what it is told is free, so a stale inventory under-allocates rather
    than double-booking a device somebody is using.
    """
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload["gpus"] if isinstance(payload, dict) else payload
    return [
        GpuCandidate(
            gpu_id=str(entry["gpu_id"]),
            free_gpu_minutes=float(entry["free_gpu_minutes"]),
            trusted=bool(entry.get("trusted", True)),
        )
        for entry in entries
    ]


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
        help="JSON GPU inventory: {gpus: [{gpu_id, free_gpu_minutes, trusted}]}.",
    )
    parser.add_argument(
        "--measured",
        type=Path,
        help="JSON {stage: minutes} from completed runs; overrides the estimate.",
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

    plan = build_model_version_plan(
        spec.model_dump(mode="json"),
        pool=_inventory(args.inventory),
        budget_gpu_minutes=args.budget_gpu_minutes,
        readiness=readiness,
        measured=_measured(args.measured),
    )

    encoded = json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)

    # Non-zero when nothing was planned, so a shell can gate on it without
    # parsing the document. The plan is still printed either way.
    return 0 if plan.within_budget else 2


if __name__ == "__main__":
    sys.exit(main())
