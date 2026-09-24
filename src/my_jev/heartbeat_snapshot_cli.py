from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .heartbeat_snapshot import (
    AuthorityContext,
    FleetStateSummary,
    KnowledgeStateSummary,
    WorkStateSummary,
    build_heartbeat_snapshot,
    snapshot_sha256,
)


def _read(path: Path):
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
        description="Build a bounded Hermes heartbeat snapshot from host-projected JSON."
    )
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--knowledge", type=Path, required=True)
    parser.add_argument("--fleet", type=Path, required=True)
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--capability", action="append", default=[])
    parser.add_argument("--tool", action="append", default=[])
    parser.add_argument("--observed-at")
    parser.add_argument("--ttl-seconds", type=int, default=300)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    work = WorkStateSummary.model_validate(_read(args.work))
    knowledge = KnowledgeStateSummary.model_validate(_read(args.knowledge))
    fleet = FleetStateSummary.model_validate(_read(args.fleet))
    authority = (
        AuthorityContext.model_validate(_read(args.authority))
        if args.authority
        else AuthorityContext()
    )
    metadata = _read(args.metadata) if args.metadata else {}
    if not isinstance(metadata, dict):
        raise ValueError("--metadata must contain a JSON object")

    snapshot = build_heartbeat_snapshot(
        work=work,
        knowledge=knowledge,
        fleet=fleet,
        authority_context=authority,
        available_capabilities=args.capability,
        available_tools=args.tool,
        metadata={str(k): str(v) for k, v in metadata.items()},
        observed_at=_time(args.observed_at),
        ttl_seconds=args.ttl_seconds,
    )
    rendered = json.dumps(snapshot.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")

    print(
        json.dumps(
            {
                "schema_version": snapshot.schema_version,
                "snapshot_sha256": snapshot_sha256(snapshot),
                "knowledge_revision": snapshot.knowledge.knowledge_revision,
                "neo4j_snapshot_id": snapshot.knowledge.neo4j_snapshot_id,
                "fleet_projection_checksum": snapshot.fleet.projection_checksum,
                "evidence_only": True,
                "runtime_authority_changed": False,
                "output": str(args.output) if args.output else None,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
