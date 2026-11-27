"""Offline contract over the committed files in ``contracts/``.

This is a producer/consumer guard with no runtime surface: the JSON schema in
``contracts/`` is a committed artifact that downstream consumers read by path.
If it is dropped, renamed, or quietly opened up (``additionalProperties``
flipped, required fields removed), nothing in the Python package fails at
import time -- the consumer simply loses the closed-shape guarantee.

Everything below is stdlib-only and reads local, committed files. There is no
network call, no credential or environment lookup, no service, no DB, no model
load and no GPU work. The ``$id``/``$schema`` strings are treated as literal
identifiers, never fetched.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_DIR = REPO_ROOT / "contracts"
HEARTBEAT_SCHEMA = CONTRACTS_DIR / "hermes-system-one-heartbeat-v1.schema.json"

# Files a consumer may reasonably expect to find in a fresh checkout.
COMMITTED_CONTRACT_FILES = (
    HEARTBEAT_SCHEMA,
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "README.md",
)

EXPECTED_SCHEMA_ID = (
    "https://scottjoyner.github.io/contracts/hermes-system-one-heartbeat-v1.schema.json"
)
EXPECTED_DIALECT = "https://json-schema.org/draft/2020-12/schema"
EXPECTED_REQUIRED = frozenset(
    {
        "profile",
        "uhp_version",
        "contract_sha256",
        "receipt_id",
        "observed_at",
        "expires_at",
        "binding",
        "advice",
        "authority",
        "provenance",
    }
)


def _load_json(path: Path) -> dict:
    """Parse a committed JSON file from disk; no resolver, no fetch."""
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict), f"{path.relative_to(REPO_ROOT)} is not a JSON object"
    return value


def test_committed_contract_files_are_present_and_nonempty() -> None:
    for path in COMMITTED_CONTRACT_FILES:
        assert path.is_file(), f"missing committed file: {path.relative_to(REPO_ROOT)}"
        assert path.stat().st_size > 0, f"empty committed file: {path.relative_to(REPO_ROOT)}"


def test_every_committed_schema_file_parses_offline() -> None:
    schemas = sorted(CONTRACTS_DIR.glob("*.schema.json"))
    assert schemas, "contracts/ has no committed *.schema.json files"
    for path in schemas:
        value = _load_json(path)
        assert isinstance(value.get("$schema"), str), f"{path.name}: no $schema dialect"
        assert isinstance(value.get("$id"), str), f"{path.name}: no $id identifier"
        assert isinstance(value.get("properties"), dict), f"{path.name}: no properties map"


def test_heartbeat_schema_identity_and_closed_shape_are_pinned() -> None:
    schema = _load_json(HEARTBEAT_SCHEMA)
    assert schema["$schema"] == EXPECTED_DIALECT
    assert schema["$id"] == EXPECTED_SCHEMA_ID
    assert schema["additionalProperties"] is False, (
        "the heartbeat schema must stay closed to unknown properties"
    )
    assert schema["type"] == "object"


def test_heartbeat_required_fields_cover_every_property() -> None:
    schema = _load_json(HEARTBEAT_SCHEMA)
    required = frozenset(schema["required"])
    properties = frozenset(schema["properties"])
    assert required == EXPECTED_REQUIRED
    assert required == properties, (
        "every declared property must be required, or an omitted field would pass"
    )
