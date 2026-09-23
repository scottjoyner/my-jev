from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from .agent_policy import AgentPolicyState, PolicyConstraints
from .server import AgentPolicyRequest, PolicyRuntime

_CORRECTION_FIELDS = (
    "user_correction",
    "operator_correction",
    "correction",
    "corrected",
    "contradicted",
    "undone",
    "user_undid",
    "user_contradicted",
    "verification_failed",
    "outcome_failed",
)


class ShadowPolicyRuntime(Protocol):
    checkpoint: str

    def agent_policy(
        self,
        request: AgentPolicyRequest,
    ) -> dict[str, object]: ...


def _object(value: Any, *, field: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"{field} must contain a JSON object")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _correction_evidence_fields(
    row: dict[str, Any],
    evidence: dict[str, Any],
) -> list[str]:
    fields: set[str] = set()
    for source in (row, evidence):
        for field in _CORRECTION_FIELDS:
            value = source.get(field)
            if value is None or value is False:
                continue
            if isinstance(
                value,
                (str, list, dict, tuple, set),
            ) and not value:
                continue
            fields.add(field)
    return sorted(fields)


def replay_row(
    row: dict[str, Any],
    runtime: ShadowPolicyRuntime,
) -> dict[str, Any]:
    """Score one captured AssistX request without dispatching its result."""
    evidence = _object(
        row.get("policy_shadow_json"),
        field="policy_shadow_json",
    )
    request = _object(
        evidence.get("request"),
        field="policy shadow request",
    )
    state = AgentPolicyState.model_validate(
        _object(
            request.get("state"),
            field="policy shadow request.state",
        )
    )
    constraints = PolicyConstraints.model_validate(
        request.get("constraints") or {}
    )

    prediction = runtime.agent_policy(
        AgentPolicyRequest(
            state=state,
            constraints=constraints,
        )
    )
    resolved = prediction.get("resolved")
    if not isinstance(resolved, dict):
        raise ValueError("runtime response missing resolved object")
    assistx = prediction.get("assistx")
    if not isinstance(assistx, dict):
        raise ValueError("runtime response missing assistx object")

    legacy = evidence.get("legacy")
    if not isinstance(legacy, dict):
        legacy = {}
    previous = evidence.get("response")
    if not isinstance(previous, dict):
        previous = {}
    previous_resolved = previous.get("resolved")
    if not isinstance(previous_resolved, dict):
        previous_resolved = {}

    return {
        "schema_version": 1,
        "mode": "shadow_replay",
        "dispatch_allowed": False,
        "intent_id": str(
            row.get("intent_id")
            or state.metadata.get("assistx_intent_id", "")
        ),
        "source": str(row.get("source") or state.source),
        "checkpoint": str(prediction.get("checkpoint") or runtime.checkpoint),
        "temperature": prediction.get("temperature"),
        "request": {
            "state": state.model_dump(mode="json"),
            "constraints": constraints.model_dump(mode="json"),
        },
        "correction_evidence_fields": (
            _correction_evidence_fields(
                row,
                evidence,
            )
        ),
        "legacy": {
            "classification": str(
                row.get("legacy_classification")
                or legacy.get("classification", "")
            ),
            "policy_action": str(
                row.get("legacy_policy_action")
                or legacy.get("policy_action", "")
            ),
        },
        "previous_shadow": {
            "model_route": str(
                previous_resolved.get("model_route", "")
            ),
            "disposition": str(
                previous_resolved.get("disposition", "")
            ),
        },
        "candidate": {
            "model_route": str(resolved.get("model_route", "")),
            "disposition": str(resolved.get("disposition", "")),
            "classification": str(
                assistx.get("classification", "")
            ),
            "policy_action": str(
                assistx.get("policy_action", "")
            ),
            "consistency_violations": list(
                resolved.get("consistency_violations") or []
            ),
        },
        "candidate_response": prediction,
    }


def replay_export(
    *,
    input_path: str | Path,
    output_path: str | Path,
    runtime: ShadowPolicyRuntime,
) -> dict[str, Any]:
    source = Path(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    disposition_counts: dict[str, int] = {}
    legacy_action_agreements = 0
    previous_disposition_agreements = 0
    previous_comparable = 0
    consistency_violations = 0

    with source.open("r", encoding="utf-8") as handle, output.open(
        "w", encoding="utf-8"
    ) as sink:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row is not a JSON object")
                replay = replay_row(row, runtime)
            except Exception as exc:
                raise ValueError(
                    f"invalid shadow row {source}:{line_number}: {exc}"
                ) from exc

            sink.write(
                json.dumps(replay, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            count += 1
            candidate = replay["candidate"]
            disposition = str(candidate["disposition"])
            disposition_counts[disposition] = (
                disposition_counts.get(disposition, 0) + 1
            )
            if (
                candidate["policy_action"]
                == replay["legacy"]["policy_action"]
            ):
                legacy_action_agreements += 1
            previous = replay["previous_shadow"]["disposition"]
            if previous:
                previous_comparable += 1
                if disposition == previous:
                    previous_disposition_agreements += 1
            if candidate["consistency_violations"]:
                consistency_violations += 1

    if count == 0:
        raise ValueError(f"no shadow records in {source}")

    manifest = {
        "schema_version": 1,
        "mode": "shadow_replay",
        "dispatch_allowed": False,
        "checkpoint": runtime.checkpoint,
        "records": count,
        "input": str(source),
        "input_sha256": _sha256(source),
        "output": str(output),
        "output_sha256": _sha256(output),
        "summary": {
            "dispositions": dict(sorted(disposition_counts.items())),
            "legacy_policy_action_agreement_rate": (
                legacy_action_agreements / count
            ),
            "previous_shadow_comparable_records": previous_comparable,
            "previous_shadow_disposition_agreement_rate": (
                previous_disposition_agreements / previous_comparable
                if previous_comparable
                else None
            ),
            "records_with_consistency_violations": consistency_violations,
            "consistency_violation_record_rate": (
                consistency_violations / count
            ),
        },
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay captured AssistX policy states through a checkpoint "
            "without dispatching or mutating Hermes/AssistX state"
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()

    runtime = PolicyRuntime.load(
        args.checkpoint,
        calibration=args.calibration,
        device=args.device,
    )
    manifest = replay_export(
        input_path=args.input,
        output_path=args.output,
        runtime=runtime,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
