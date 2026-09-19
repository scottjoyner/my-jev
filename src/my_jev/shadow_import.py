from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .agent_policy import (
    AgentPolicyState,
    build_agent_policy_record,
)
from .data import dump_jsonl
from .manifest import write_manifest
from .schema import DecisionRecord

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


def _json_object(
    value: Any,
    *,
    field: str,
) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(
        f"{field} must contain a JSON object"
    )


def _dict_or_empty(
    value: Any,
) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_of_strings(
    value: Any,
) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        str(item)
        for item in value
        if str(item).strip()
    ]


def _has_evidence(
    value: Any,
) -> bool:
    if value is None or value is False:
        return False
    if isinstance(
        value,
        (str, list, dict, tuple, set),
    ):
        return bool(value)
    return True


def _correction_evidence_fields(
    row: dict[str, Any],
    evidence: dict[str, Any],
) -> list[str]:
    fields = {
        field
        for source in (row, evidence)
        for field in _CORRECTION_FIELDS
        if field in source
        and _has_evidence(source[field])
    }
    return sorted(fields)


def shadow_row_to_record(
    row: dict[str, Any],
) -> DecisionRecord:
    """Convert one AssistX shadow-export row into an unlabeled policy record.

    Model output and legacy routing remain evidence metadata only. They are
    intentionally never converted to targets.
    """
    evidence = _json_object(
        row.get("policy_shadow_json"),
        field="policy_shadow_json",
    )
    request = _json_object(
        evidence.get("request"),
        field="policy shadow request",
    )
    state_payload = _json_object(
        request.get("state"),
        field="policy shadow request.state",
    )
    state = AgentPolicyState.model_validate(
        state_payload
    )

    response = _dict_or_empty(
        evidence.get("response")
    )
    resolved = _dict_or_empty(
        response.get("resolved")
    )
    assistx = _dict_or_empty(
        response.get("assistx")
    )
    scores = _dict_or_empty(
        response.get("scores")
    )

    record = build_agent_policy_record(
        state
    )
    intent_id = str(
        row.get("intent_id")
        or state.metadata.get(
            "assistx_intent_id",
            "",
        )
    )
    tasks = row.get("created_tasks")
    if not isinstance(tasks, list):
        tasks = []

    route_confidence = resolved.get(
        "model_route_confidence"
    )
    if not isinstance(
        route_confidence,
        (int, float),
    ):
        route_confidence = None

    record.metadata.update(
        {
            "domain": "assistx_agent_policy_shadow",
            "family_id": intent_id or None,
            "source_intent_id": intent_id,
            "source": str(
                row.get("source")
                or state.source
            ),
            "legacy_classification": str(
                row.get(
                    "legacy_classification"
                )
                or _dict_or_empty(
                    evidence.get("legacy")
                ).get(
                    "classification",
                    "",
                )
            ),
            "legacy_policy_action": str(
                row.get(
                    "legacy_policy_action"
                )
                or _dict_or_empty(
                    evidence.get("legacy")
                ).get(
                    "policy_action",
                    "",
                )
            ),
            "shadow_checkpoint": str(
                response.get(
                    "checkpoint",
                    "",
                )
            ),
            "shadow_temperature": (
                response.get("temperature")
            ),
            "shadow_scores": scores,
            "shadow_model_route": str(
                resolved.get(
                    "model_route",
                    "",
                )
            ),
            "shadow_model_route_confidence": (
                route_confidence
            ),
            "shadow_disposition": str(
                resolved.get(
                    "disposition",
                    "",
                )
            ),
            "shadow_consistency_violations": (
                _list_of_strings(
                    resolved.get(
                        "consistency_violations"
                    )
                )
            ),
            "shadow_resolver_reasons": (
                _list_of_strings(
                    resolved.get("reasons")
                )
            ),
            "shadow_classification": str(
                assistx.get(
                    "classification",
                    "",
                )
            ),
            "shadow_policy_action": str(
                assistx.get(
                    "policy_action",
                    "",
                )
            ),
            "correction_evidence_fields": (
                _correction_evidence_fields(
                    row,
                    evidence,
                )
            ),
            "created_tasks": tasks,
            "redacted": bool(
                row.get(
                    "redacted",
                    False,
                )
            ),
            "label_status": "unlabeled",
            "label_source": None,
        }
    )
    return record


def load_shadow_export(
    path: str | Path,
) -> list[DecisionRecord]:
    records: list[DecisionRecord] = []
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(
                        "row is not a JSON object"
                    )
                records.append(
                    shadow_row_to_record(
                        row
                    )
                )
            except Exception as exc:
                raise ValueError(
                    f"invalid shadow row "
                    f"{path}:{line_number}: {exc}"
                ) from exc
    if not records:
        raise ValueError(
            f"no shadow records in {path}"
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an AssistX my-jev shadow export into "
            "unlabeled typed policy records"
        )
    )
    parser.add_argument(
        "--input",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    args = parser.parse_args()

    records = load_shadow_export(
        args.input
    )
    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    dump_jsonl(
        iter(records),
        output,
    )
    manifest_path = output.with_suffix(
        output.suffix
        + ".manifest.json"
    )
    manifest = write_manifest(
        output,
        manifest_path,
    )
    print(
        f"wrote {manifest['records']} unlabeled policy states, "
        f"sha256={manifest['sha256']}"
    )
    print(
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
