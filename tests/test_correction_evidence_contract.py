"""Correction evidence is a producer/consumer contract across three modules.

`shadow_replay` and `shadow_import` write which fields of a record show a human
corrected it; `review.has_correction_evidence` reads that key and turns it into a
share of a record's review priority. Rename the key on one side and nothing fails --
the consumer falls through to the older `user_corrected` / `operator_corrected`
booleans, which the replay importer never writes, and a human-corrected record
silently stops counting as corrected.

The field *list* was equally unpinned: of the ten field names
`_CORRECTION_FIELDS` recognises, only `user_correction` appeared anywhere in the
test suite. Dropping one from that tuple would have been silent too.
"""

from __future__ import annotations

import pytest

from my_jev.review import has_correction_evidence
from my_jev.model import QuestionType
from my_jev.schema import (
    CORRECTION_EVIDENCE_FIELDS,
    DecisionRecord,
    QuestionSpec,
)
from my_jev.shadow_replay import _CORRECTION_FIELDS, _correction_evidence_fields


def _record(metadata: dict) -> DecisionRecord:
    return DecisionRecord(
        state="a node was unavailable",
        questions={
            "placement": QuestionSpec(
                type=QuestionType.CHOICE,
                instructions="Where should this run?",
                options=["local", "defer"],
            )
        },
        metadata=metadata,
    )


def _produced_by_replay(row: dict) -> dict:
    """The metadata the replay importer would write for one row."""
    fields = _correction_evidence_fields(row, {})
    return {CORRECTION_EVIDENCE_FIELDS: fields}


# --- the key ---------------------------------------------------------------


def test_the_key_is_defined_once_and_is_a_metadata_key_on_the_record():
    assert CORRECTION_EVIDENCE_FIELDS == "correction_evidence_fields"


def test_the_replay_producer_writes_exactly_the_key_the_consumer_reads():
    """The end-to-end contract, in one assertion.

    Each side used its own string literal. This is the assertion whose absence let
    them drift: it fails if either renames the key without the other.
    """
    metadata = _produced_by_replay({"user_correction": "wrong placement"})
    assert has_correction_evidence(_record(metadata)) is True


def test_the_import_producer_writes_the_same_key():
    """`shadow_import` is a second producer and had its own literal."""
    import ast
    import pathlib

    source = pathlib.Path("src/my_jev/shadow_import.py").read_text()
    assigned = [
        key.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant)
        for key in ()
    ]
    del assigned
    # `shadow_import` is a second producer with its own literal. Assert it now uses
    # the shared constant, by checking the module namespace exposes it -- a source
    # read would pass on the old code, which merely had the string inline.
    module = __import__("my_jev.shadow_import", fromlist=["CORRECTION_EVIDENCE_FIELDS"])
    assert module.CORRECTION_EVIDENCE_FIELDS == CORRECTION_EVIDENCE_FIELDS


# --- the field list --------------------------------------------------------


#: Spelled out rather than imported, on purpose.
#:
#: The obvious version of the test below is
#: `@pytest.mark.parametrize("field", _CORRECTION_FIELDS)`, which iterates whatever
#: the list happens to contain -- so deleting a field deleted its own test case and
#: the suite stayed green. Mutation testing caught exactly that. A contract list
#: has to be written down independently somewhere or it is not pinned at all.
EXPECTED_CORRECTION_FIELDS = (
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


def test_the_recognised_field_list_is_exactly_what_is_expected():
    """Written down independently, so a silent deletion cannot pass.

    The failure direction matters: a field dropped from this tuple stops that kind
    of correction counting, so a human-corrected record is queued as ordinary.
    """
    assert tuple(_CORRECTION_FIELDS) == EXPECTED_CORRECTION_FIELDS


@pytest.mark.parametrize("field", EXPECTED_CORRECTION_FIELDS)
def test_every_recognised_correction_field_is_actually_recognised(field):
    """All ten, not the one the suite happened to mention.

    A field dropped from `_CORRECTION_FIELDS` would stop that kind of correction
    counting, lowering review priority for exactly the records a human corrected --
    with no other symptom.
    """
    metadata = _produced_by_replay({field: True})
    assert metadata[CORRECTION_EVIDENCE_FIELDS] == [field]
    assert has_correction_evidence(_record(metadata)) is True


def test_a_field_absent_from_the_list_is_not_treated_as_correction():
    """And the list is not merely decorative.

    Without this, a test that every listed field works would still pass if the
    helper counted every field it was handed.
    """
    metadata = _produced_by_replay({"some_unrelated_field": True})
    assert metadata[CORRECTION_EVIDENCE_FIELDS] == []
    assert has_correction_evidence(_record(metadata)) is False


def test_falsy_values_do_not_count_as_correction():
    """A field present but falsy is not evidence of a correction.

    `shadow_replay` writes `False` for absent booleans on most records, so counting
    presence rather than truth would mark the entire corpus as corrected.

    `0` and `0.0` are in this list because the check used to be `value is None or
    value is False` -- an identity comparison, so a producer emitting `0` for an
    unset flag had the record counted as human-corrected.
    """
    for value in (False, None, 0, 0.0, "", [], {}):
        metadata = _produced_by_replay({"user_correction": value})
        assert metadata[CORRECTION_EVIDENCE_FIELDS] == [], value
        assert has_correction_evidence(_record(metadata)) is False


def test_correction_evidence_is_deduplicated_across_both_sources():
    """`_correction_evidence_fields` scans the row *and* the evidence blob."""
    fields = _correction_evidence_fields(
        {"user_correction": True},
        {"user_correction": True},
    )
    assert fields == ["user_correction"]


# --- the fallback path -----------------------------------------------------


def test_the_boolean_fallback_still_works_for_records_without_the_list():
    """Pre-existing records carry the older booleans and must not be ignored."""
    assert has_correction_evidence(
        _record({"user_corrected": True})
    ) is True
    assert has_correction_evidence(
        _record({"operator_corrected": True})
    ) is True


def test_a_record_with_no_evidence_at_all_is_not_a_correction():
    assert has_correction_evidence(_record({})) is False
