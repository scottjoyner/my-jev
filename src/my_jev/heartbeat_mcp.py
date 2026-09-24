from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from .heartbeat_snapshot import (
    HeartbeatSnapshot,
    canonical_snapshot_json,
    snapshot_sha256,
)

MODE_OPTIONS = (
    "chat",
    "create_tasks",
    "act",
    "clarify",
    "cancel",
    "abstain",
)
NONE = "none"
MAX_CLOCK_SKEW = timedelta(minutes=5)

INSTRUCTIONS = (
    "This environment is advisory only. Observe the bounded Hermes heartbeat snapshot, "
    "then choose exactly one recommendation. A recommendation never grants dispatch, "
    "approval, claims, mutation, tool access, or routing authority. Fleet values are "
    "opaque handles from an already-authoritative eligible set. Do not infer endpoints, "
    "credentials, providers, or additional capabilities from them."
)


class HeartbeatAdvisoryEnvironment:
    """Finite, no-external-effect System-One environment over one heartbeat snapshot."""

    def __init__(self, snapshot_path: str | Path):
        self.snapshot_path = Path(snapshot_path)
        self.recommendation: dict[str, object] | None = None
        self.snapshot = self._load()

    def _load(self, *, now: datetime | None = None) -> HeartbeatSnapshot:
        try:
            snapshot = HeartbeatSnapshot.model_validate_json(
                self.snapshot_path.read_text(encoding="utf-8")
            )
        except OSError as exc:
            raise RuntimeError(f"heartbeat snapshot could not be read: {exc}") from exc
        self._assert_fresh(snapshot, now=now)
        return snapshot

    @staticmethod
    def _parse_stamp(value: str) -> datetime:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("heartbeat timestamp must be timezone-aware")
        return parsed.astimezone(UTC)

    @classmethod
    def _assert_fresh(
        cls,
        snapshot: HeartbeatSnapshot,
        *,
        now: datetime | None = None,
    ) -> None:
        current = (now or datetime.now(UTC)).astimezone(UTC)
        observed = cls._parse_stamp(snapshot.observed_at)
        expires = cls._parse_stamp(snapshot.expires_at)
        if observed > current + MAX_CLOCK_SKEW:
            raise RuntimeError("heartbeat snapshot observation is in the future")
        if expires <= current:
            raise RuntimeError("heartbeat snapshot is expired")
        if expires <= observed:
            raise RuntimeError("heartbeat snapshot has an invalid TTL")

    def reset(self, goal: str = "") -> dict[str, object]:
        del goal
        self.recommendation = None
        self.snapshot = self._load()
        return {
            "ok": True,
            "text": "Advisory heartbeat episode reset.",
            "fields": {
                "snapshot_sha256": snapshot_sha256(self.snapshot),
            },
        }

    def observe(self, *, now: datetime | None = None) -> dict[str, object]:
        self.snapshot = self._load(now=now)
        terminal = self.recommendation is not None
        return {
            "text": canonical_snapshot_json(self.snapshot),
            "fields": {
                "schema_version": self.snapshot.schema_version,
                "snapshot_sha256": snapshot_sha256(self.snapshot),
                "work_id": self.snapshot.work.work_id,
                "work_status": self.snapshot.work.status,
                "knowledge_revision": self.snapshot.knowledge.knowledge_revision,
                "neo4j_snapshot_id": self.snapshot.knowledge.neo4j_snapshot_id,
                "fleet_projection_generation": self.snapshot.fleet.projection_generation,
                "fleet_projection_checksum": self.snapshot.fleet.projection_checksum,
                "fleet_observation_snapshot_id": self.snapshot.fleet.observation_snapshot_id,
                "fleet_pressure": self.snapshot.fleet.pressure,
                "recommendation": self.recommendation,
                "evidence_only": True,
                "runtime_authority_changed": False,
            },
            "candidates": {
                "eligible_fleet_handles": [
                    NONE,
                    *self.snapshot.fleet.eligible_handles,
                ],
                "context_focus": [
                    NONE,
                    *self.snapshot.knowledge.note_refs,
                ],
            },
            "terminal": terminal,
            "artifacts": [],
            "realtime": False,
        }

    def recommend(
        self,
        *,
        mode: str,
        fleet_handle: str,
        context_focus: str,
    ) -> dict[str, object]:
        if mode not in MODE_OPTIONS:
            raise ValueError("mode is outside the finite advisory vocabulary")
        fleet_candidates = {NONE, *self.snapshot.fleet.eligible_handles}
        if fleet_handle not in fleet_candidates:
            raise ValueError("fleet_handle is outside the authoritative eligible set")
        context_candidates = {NONE, *self.snapshot.knowledge.note_refs}
        if context_focus not in context_candidates:
            raise ValueError("context_focus is outside the bounded snapshot candidates")

        recommendation = {
            "mode": mode,
            "fleet_handle": None if fleet_handle == NONE else fleet_handle,
            "context_focus": None if context_focus == NONE else context_focus,
        }
        authority = {
            "dispatch_allowed": False,
            "approval_granted": False,
            "claim_acquired": False,
            "mutation_allowed": False,
            "routing_authority_changed": False,
        }
        envelope = {
            "schema": "hermes-system-one-recommendation-v1",
            "snapshot_sha256": snapshot_sha256(self.snapshot),
            "advice": recommendation,
            "authority": authority,
            "evidence_only": True,
            "runtime_authority_changed": False,
        }
        rendered = json.dumps(
            envelope,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.recommendation = recommendation
        return {
            "ok": True,
            "text": rendered,
            "fields": envelope,
            "artifacts": [
                f"hermes-system-one-recommendation.json: {rendered}",
            ],
            "terminal": True,
        }


def build(snapshot_path: str | Path):
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.types import ToolAnnotations
    except ImportError as exc:
        raise RuntimeError(
            "MCP support requires the optional systemone extra: "
            "pip install 'my-jev[systemone]'"
        ) from exc

    env = HeartbeatAdvisoryEnvironment(snapshot_path)
    server = MCPServer(
        "hermes-system-one-heartbeat",
        instructions=INSTRUCTIONS,
        log_level="WARNING",
    )

    @server.tool(
        annotations=ToolAnnotations(readOnlyHint=True),
        description="Observe the bounded, credential-free Hermes heartbeat snapshot.",
    )
    def observe() -> dict:
        return env.observe()

    @server.tool(
        annotations=ToolAnnotations(readOnlyHint=True),
        description="Reset only this in-memory advisory episode and reload the snapshot.",
    )
    def reset(goal: str = "") -> dict:
        return env.reset(goal)

    @server.tool(
        annotations=ToolAnnotations(readOnlyHint=True),
        description=(
            "Record one advisory recommendation and terminate the episode. "
            "This has no external side effect and grants no authority."
        ),
    )
    def recommend(
        mode: Literal[
            "chat",
            "create_tasks",
            "act",
            "clarify",
            "cancel",
            "abstain",
        ],
        fleet_handle: Annotated[
            str,
            Field(
                description="Opaque fleet preference or none.",
                json_schema_extra={"x-candidates": "eligible_fleet_handles"},
            ),
        ],
        context_focus: Annotated[
            str,
            Field(
                description="Bounded knowledge note to prioritize or none.",
                json_schema_extra={"x-candidates": "context_focus"},
            ),
        ],
    ) -> dict:
        return env.recommend(
            mode=mode,
            fleet_handle=fleet_handle,
            context_focus=context_focus,
        )

    return server, env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the bounded Hermes heartbeat as a finite System-One MCP environment."
    )
    parser.add_argument(
        "--snapshot",
        default=os.environ.get("HERMES_HEARTBEAT_SNAPSHOT", ""),
        help="Path to hermes-heartbeat-snapshot-v1 JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.snapshot:
        print(
            "--snapshot or HERMES_HEARTBEAT_SNAPSHOT is required",
            file=sys.stderr,
        )
        return 2
    try:
        server, _ = build(args.snapshot)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    server.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
