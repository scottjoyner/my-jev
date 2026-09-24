from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .agent_policy import ResolvedAgentPolicy
from .fleet_resolver import FleetPlacementResolution
from .uhp_advisory import (
    SystemOneProvenance,
    build_hermes_system_one_profile,
    build_uhp_response_fixture,
    canonical_sha256,
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a deterministic Local Studio System-One acceptance fixture suite."
    )
    parser.add_argument("--decision", type=Path, default=Path("examples/uhp/decision.json"))
    parser.add_argument(
        "--fleet-resolution",
        type=Path,
        default=Path("examples/uhp/fleet-resolution.json"),
    )
    parser.add_argument(
        "--fleet-handle-map",
        type=Path,
        default=Path("examples/uhp/fleet-handles.json"),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("examples/uhp/provenance.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--now")
    return parser


def _now(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc).replace(microsecond=0)
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("--now must include an offset or Z")
    return parsed.astimezone(timezone.utc).replace(microsecond=0)


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    now = _now(args.now)
    decision = ResolvedAgentPolicy.model_validate(_load(args.decision))
    fleet = FleetPlacementResolution.model_validate(_load(args.fleet_resolution))
    handles = {str(k): str(v) for k, v in _load(args.fleet_handle_map).items()}
    provenance = SystemOneProvenance.model_validate(_load(args.provenance))

    profile = build_hermes_system_one_profile(
        decision,
        receipt_id="acceptance-valid",
        observed_at=now,
        ttl_seconds=600,
        task_focus="Acceptance fixture: advisory context only.",
        context_priority=["current-pr", "latest-handoff"],
        fleet_resolution=fleet,
        fleet_handle_by_node_id=handles,
        provenance=provenance,
    )
    valid = build_uhp_response_fixture(
        profile,
        response_id="resp_acceptance_valid",
        session_id="hsess-acceptance",
        harness_id="chrn_system_one",
        model="recorded/jev",
        created_at=now,
    )

    expired = copy.deepcopy(valid)
    expired["id"] = "resp_acceptance_expired"
    expired_profile = expired["metadata"]["hermes_system_one"]
    expired_profile["receipt_id"] = "acceptance-expired"
    expired_profile["observed_at"] = (
        now - timedelta(minutes=20)
    ).isoformat(timespec="seconds").replace("+00:00", "Z")
    expired_profile["expires_at"] = (
        now - timedelta(minutes=10)
    ).isoformat(timespec="seconds").replace("+00:00", "Z")

    authority = copy.deepcopy(valid)
    authority["id"] = "resp_acceptance_authority"
    authority_profile = authority["metadata"]["hermes_system_one"]
    authority_profile["receipt_id"] = "acceptance-authority"
    authority_profile["authority"]["mutation_allowed"] = True

    fallback = copy.deepcopy(valid)
    fallback["id"] = "resp_acceptance_fallback"
    fallback_profile = fallback["metadata"]["hermes_system_one"]
    fallback_profile["receipt_id"] = "acceptance-fallback"
    fallback["metadata"]["requested_model"] = "requested/jev"
    fallback["metadata"]["model_fallback"] = True
    fallback["metadata"]["model_fallback_reason"] = "acceptance fixture"

    handoff = copy.deepcopy(valid)
    handoff["id"] = "resp_acceptance_handoff"
    handoff_profile = handoff["metadata"]["hermes_system_one"]
    handoff_profile["receipt_id"] = "acceptance-handoff"
    handoff["status"] = "incomplete"
    handoff["incomplete_details"] = {
        "reason": "no_confident_action",
        "handoff": {
            "state": "fixture-state",
            "weakest": 0.41,
            "threshold": 0.60,
            "risk_class": "moderate",
        },
    }

    cases = {
        "valid.json": (valid, "consumed"),
        "expired.json": (expired, "expired"),
        "authority-bearing.json": (authority, "authority_mutation_allowed"),
        "model-fallback.json": (fallback, "model_fallback"),
        "handoff.json": (handoff, "system_one_handoff"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "local-studio-system-one-acceptance-v1",
        "generated_at": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "evidence_only": True,
        "runtime_authority_changed": False,
        "cases": {},
    }
    for filename, (payload, expected) in cases.items():
        _write(args.output_dir / filename, payload)
        manifest["cases"][filename] = {
            "expected": expected,
            "sha256": canonical_sha256(payload),
        }

    _write(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
