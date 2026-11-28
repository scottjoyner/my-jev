import json

import pytest

from my_jev.data import (
    DecisionDataset,
    collate_records,
    dump_jsonl,
    load_jsonl,
)
from my_jev.schema import (
    DecisionRecord,
    QuestionSpec,
    QuestionType,
    TargetSpec,
)


def _record(
    state: str = "The agent should respond.",
    *,
    question_type: QuestionType = QuestionType.CHOICE,
    options: list[str] | None = None,
    answer_index: int = 0,
    metadata: dict | None = None,
) -> DecisionRecord:
    options = options or ["left", "right"]
    return DecisionRecord(
        state=state,
        questions={
            "decision": QuestionSpec(
                type=question_type,
                instructions="Pick one.",
                options=options,
            )
        },
        targets={"decision": TargetSpec(index=answer_index)},
        metadata=metadata or {},
    )


def test_dump_load_jsonl_round_trip(tmp_path):
    records = [
        _record("État initial — naïve façade", answer_index=0),
        _record(
            "second state",
            question_type=QuestionType.NOUL,
            options=["false", "true"],
            answer_index=1,
            metadata={"family_id": "f-2", "nested": {"ok": True}},
        ),
    ]
    path = tmp_path / "records.jsonl"

    dump_jsonl(iter(records), path)

    text = path.read_text(encoding="utf-8")
    assert "État initial" in text  # ensure_ascii=False
    assert len(text.splitlines()) == 2
    assert [r.model_dump() for r in load_jsonl(path)] == [
        r.model_dump() for r in records
    ]


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "records.jsonl"
    line = _record().model_dump_json()
    path.write_text(
        f"\n{line}\n   \n\t\n{line}\n",
        encoding="utf-8",
    )

    assert len(load_jsonl(path)) == 2


def test_load_jsonl_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")

    assert load_jsonl(path) == []


def test_load_jsonl_reports_physical_line_of_invalid_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        "\n".join(
            [
                _record().model_dump_json(),
                "",
                "{not valid json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=rf"invalid record at {path}:3:",
    ):
        load_jsonl(path)


def test_load_jsonl_wraps_schema_validation_errors(tmp_path):
    path = tmp_path / "schema_bad.jsonl"
    bad = _record().model_dump(mode="json")
    bad["targets"]["decision"]["index"] = 99
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid record at") as info:
        load_jsonl(path)

    assert "target index out of range" in str(info.value)
    assert info.value.__cause__ is not None


def test_decision_dataset_indexing_and_collate_identity(tmp_path):
    path = tmp_path / "records.jsonl"
    records = [_record("a"), _record("b")]
    dump_jsonl(iter(records), path)

    dataset = DecisionDataset(path)

    assert len(dataset) == 2
    assert dataset[1] == records[1]
    batch = [dataset[0], dataset[1]]
    assert collate_records(batch) is batch
