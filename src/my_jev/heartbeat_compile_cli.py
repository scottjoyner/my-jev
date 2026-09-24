from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from .heartbeat_compile import TerminalRecommendation, compile_heartbeat_recommendation
from .heartbeat_snapshot import HeartbeatSnapshot, snapshot_sha256
from .uhp_advisory import (
    DEFAULT_TTL_SECONDS,
    SystemOneProvenance,
    build_uhp_response_fixture,
    canonical_sha256,
)


def _time(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamps must include an offset or Z")
    return parsed


def _sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile one terminal System-One heartbeat recommendation into a "
            "session/project-bound stored UHP response."
        )
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--recommendation", type=Path, required=True)
    parser.add_argument("--receipt-id", required=True)
    parser.add_argument("--consumer-session-id", required=True)
    parser.add_argument("--project-cwd", type=Path, required=True)
    parser.add_argument("--consumer", default="local-studio")
    parser.add_argument("--compiled-at")
    parser.add_argument("--ttl-seconds", type=int, default=DEFAULT_TTL_SECONDS)
    parser.add_argument("--mode-confidence", type=float, required=True)
    parser.add_argument("--policy-disposition")
    parser.add_argument("--approval-recommended", action="store_true")
    parser.add_argument("--task-focus")
    parser.add_argument("--system-one-config-version")
    parser.add_argument("--model-revision")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--response-id", required=True)
    parser.add_argument("--uhp-session-id", required=True)
    parser.add_argument("--harness-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--previous-response-id")
    parser.add_argument("--created-at")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    snapshot = HeartbeatSnapshot.model_validate_json(
        args.snapshot.read_text(encoding="utf-8")
    )
    recommendation = TerminalRecommendation.model_validate_json(
        args.recommendation.read_text(encoding="utf-8")
    )
    provenance = SystemOneProvenance(
        system_one_config_version=args.system_one_config_version,
        model_revision=args.model_revision,
        knowledge_revision=snapshot.knowledge.knowledge_revision,
        neo4j_snapshot_id=snapshot.knowledge.neo4j_snapshot_id,
        fleet_projection_generation=snapshot.fleet.projection_generation,
        fleet_projection_checksum=snapshot.fleet.projection_checksum,
        trace_sha256=_sha256_file(args.trace),
    )
    profile = compile_heartbeat_recommendation(
        snapshot,
        recommendation,
        receipt_id=args.receipt_id,
        consumer_session_id=args.consumer_session_id,
        project_cwd=args.project_cwd,
        consumer=args.consumer,
        compiled_at=_time(args.compiled_at),
        ttl_seconds=args.ttl_seconds,
        mode_confidence=args.mode_confidence,
        policy_disposition=args.policy_disposition,
        approval_recommended=(True if args.approval_recommended else None),
        task_focus=args.task_focus,
        provenance=provenance,
    )
    response = build_uhp_response_fixture(
        profile,
        response_id=args.response_id,
        session_id=args.uhp_session_id,
        harness_id=args.harness_id,
        model=args.model,
        created_at=_time(args.created_at) or _time(args.compiled_at),
        previous_response_id=args.previous_response_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(response, indent=2, sort_keys=True) + "\n"
    temp_output = args.output.with_name(f"{args.output.name}.tmp-{os.getpid()}")
    temp_output.write_text(rendered, encoding="utf-8")
    temp_output.replace(args.output)
    print(
        json.dumps(
            {
                "evidence_only": True,
                "runtime_authority_changed": False,
                "snapshot_sha256": snapshot_sha256(snapshot),
                "profile_sha256": canonical_sha256(profile),
                "response_sha256": canonical_sha256(response),
                "receipt_id": profile.receipt_id,
                "consumer_session_id": profile.binding.consumer_session_id,
                "project_fingerprint": profile.binding.project_fingerprint,
                "expires_at": profile.expires_at,
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
