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


def shadow_row_to_record(
    row: dict[str, Any],
) -> DecisionRecord:
    """Convert one AssistX shadow-export row into an unlabeled policy record.

    The shadow model output and legacy classifier are retained only as evidence
    metadata. They are intentionally NOT converted to targets.
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

    response = evidence.get("response")
    if not isinstance(response, dict):
        response = {}
    resolved = response.get("resolved")
    if not isinstance(resolved, dict):
        resolved = {}
    assistx = response.get("assistx")
    if not isinstance(assistx, dict):
        assistx = {}

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
                or evidence.get(
                    "legacy",
                    {},
                ).get(
                    "classification",
                    "",
                )
            ),
            "legacy_policy_action": str(
                row.get(
                    "legacy_policy_action"
                )
                or evidence.get(
                    "legacy",
                    {},
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
            "shadow_model_route": str(
                resolved.get(
                    "model_route",
                    "",
                )
            ),
            "shadow_disposition": str(
                resolved.get(
                    "disposition",
                    "",
                )
            ),
            "shadow_policy_action": str(
                assistx.get(
                    "policy_action",
                    "",
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
