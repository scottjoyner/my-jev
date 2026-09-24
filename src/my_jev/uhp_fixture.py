from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .agent_policy import ResolvedAgentPolicy
from .fleet_resolver import FleetPlacementResolution
from .uhp_advisory import (
    DEFAULT_TTL_SECONDS,
    SystemOneProvenance,
    build_hermes_system_one_profile,
    build_uhp_response_fixture,
    canonical_sha256,
)


def _read_json(path: Path | None) -> Any:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _time(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamps must include an offset or Z")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an advisory-only Hermes System-One profile inside a stored UHP response."
    )
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--fleet-resolution", type=Path)
    parser.add_argument("--fleet-handle-map", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--receipt-id", required=True)
    parser.add_argument("--response-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--harness-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--previous-response-id")
    parser.add_argument("--observed-at")
    parser.add_argument("--created-at")
    parser.add_argument("--ttl-seconds", type=int, default=DEFAULT_TTL_SECONDS)
    parser.add_argument("--task-focus")
    parser.add_argument("--context-priority", action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    decision = ResolvedAgentPolicy.model_validate(_read_json(args.decision))
    fleet_payload = _read_json(args.fleet_resolution)
    fleet = (
        FleetPlacementResolution.model_validate(fleet_payload)
        if fleet_payload is not None
        else None
    )
    handle_payload = _read_json(args.fleet_handle_map)
    handles = (
        {str(key): str(value) for key, value in handle_payload.items()}
        if isinstance(handle_payload, dict)
        else None
    )
    provenance_payload = _read_json(args.provenance)
    provenance = SystemOneProvenance.model_validate(provenance_payload or {})

    profile = build_hermes_system_one_profile(
        decision,
        receipt_id=args.receipt_id,
        observed_at=_time(args.observed_at),
        ttl_seconds=args.ttl_seconds,
        task_focus=args.task_focus,
        context_priority=args.context_priority,
        fleet_resolution=fleet,
        fleet_handle_by_node_id=handles,
        provenance=provenance,
    )
    response = build_uhp_response_fixture(
        profile,
        response_id=args.response_id,
        session_id=args.session_id,
        harness_id=args.harness_id,
        model=args.model,
        created_at=_time(args.created_at) or _time(args.observed_at),
        previous_response_id=args.previous_response_id,
    )
    rendered = json.dumps(response, indent=2, sort_keys=True) + "\n"

    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")

    print(
        json.dumps(
            {
                "evidence_only": True,
                "runtime_authority_changed": False,
                "response_sha256": canonical_sha256(response),
                "profile_sha256": canonical_sha256(profile),
                "output": str(args.output) if args.output else None,
            },
            sort_keys=True,
        ),
        file=__import__("sys").stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
