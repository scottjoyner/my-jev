from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from .fleet_benchmark_qualification import (
    DEFAULT_MAX_EVIDENCE_AGE_SECONDS,
    DEFAULT_MAX_HEALTH_FRESHNESS_SECONDS,
    DEFAULT_MIN_QUALITY_CONFIDENCE,
    BenchmarkWorkIntent,
    FleetBenchmarkMatrix,
    QualificationThresholds,
    build_fleet_benchmark_advisory,
)
from .fleet_policy import FleetPlacementState


def _read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _time(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("--observed-at must include an offset or Z")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Emit an advisory-only benchmark-qualification result from a fleet "
            "campaign matrix. No live AssistX access is performed."
        )
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--handle-map", type=Path, required=True)
    parser.add_argument(
        "--work-intent",
        choices=[item.value for item in BenchmarkWorkIntent],
        default=BenchmarkWorkIntent.CODING.value,
    )
    parser.add_argument("--observed-at")
    parser.add_argument(
        "--max-evidence-age-seconds",
        type=int,
        default=DEFAULT_MAX_EVIDENCE_AGE_SECONDS,
    )
    parser.add_argument(
        "--min-quality-confidence",
        type=float,
        default=DEFAULT_MIN_QUALITY_CONFIDENCE,
    )
    parser.add_argument(
        "--max-health-freshness-seconds",
        type=float,
        default=DEFAULT_MAX_HEALTH_FRESHNESS_SECONDS,
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        matrix = FleetBenchmarkMatrix.model_validate(_read(args.matrix))
        state = FleetPlacementState.model_validate(_read(args.state))
        raw_handles = _read(args.handle_map)
        if not isinstance(raw_handles, dict):
            raise ValueError("--handle-map must contain a JSON object of node_id -> handle")
        thresholds = QualificationThresholds(
            max_evidence_age_seconds=args.max_evidence_age_seconds,
            min_quality_confidence=args.min_quality_confidence,
            max_health_freshness_seconds=args.max_health_freshness_seconds,
        )
        advisory = build_fleet_benchmark_advisory(
            matrix,
            state,
            work_intent=args.work_intent,
            thresholds=thresholds,
            observed_at=_time(args.observed_at),
            handle_by_node_id={str(key): str(value) for key, value in raw_handles.items()},
        )
    except (OSError, ValueError) as error:
        print(f"my-jev-fleet-benchmark-advisory: {error}", file=sys.stderr)
        return 2

    wire = advisory.model_wire()
    rendered = json.dumps(wire, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")

    print(
        json.dumps(
            {
                "schema_version": advisory.schema_version,
                "campaign_id": advisory.campaign_id,
                "execution_shape": advisory.execution_shape.value,
                "role_assignment": (
                    advisory.role_assignment.value
                    if advisory.role_assignment is not None
                    else None
                ),
                "decomposition_fallback": (
                    advisory.decomposition_fallback.value
                    if advisory.decomposition_fallback is not None
                    else None
                ),
                "eligible_node_count": advisory.eligible_node_count,
                "preferred_handle_count": len(advisory.preferred),
                "model_state_sha256": hashlib.sha256(
                    advisory.as_model_state().encode("utf-8")
                ).hexdigest(),
                "evidence_only": True,
                "runtime_authority_changed": False,
                "assistx_accessed": False,
                "output": str(args.output) if args.output else None,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())