import pytest

from my_jev.agentic_synth import (
    generate_agent_policy_records,
)
from my_jev.data import dump_jsonl
from my_jev.data_audit import (
    audit_dataset_splits,
)
from my_jev.split import split_records


def _write_split_set(
    tmp_path,
    *,
    records=60,
):
    generated = (
        generate_agent_policy_records(
            records,
            seed=41,
        )
    )
    splits, _ = split_records(
        generated,
        train_fraction=0.7,
        validation_fraction=0.1,
        calibration_fraction=0.1,
        seed=41,
        group_key="auto",
    )
    paths = {}
    for name, split in (
        splits.items()
    ):
        path = (
            tmp_path
            / f"{name}.jsonl"
        )
        dump_jsonl(
            iter(split),
            path,
        )
        paths[name] = path
    return paths


def test_data_audit_accepts_group_safe_splits(
    tmp_path,
):
    paths = _write_split_set(
        tmp_path
    )

    report = audit_dataset_splits(
        paths
    )

    assert report["passed"] is True
    assert (
        report["group_key"]
        == "family_id"
    )
    assert (
        report[
            "semantic_overlaps"
        ]
        == []
    )
    assert (
        report["group_overlaps"]
        == []
    )


def test_data_audit_rejects_semantic_leakage(
    tmp_path,
):
    paths = _write_split_set(
        tmp_path
    )
    train = (
        paths["train"]
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )
    with paths["test"].open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            train[0] + "\n"
        )

    with pytest.raises(
        ValueError,
        match="semantic record leakage",
    ):
        audit_dataset_splits(
            paths
        )


def test_data_audit_rejects_family_leakage_even_when_state_differs(
    tmp_path,
):
    paths = _write_split_set(
        tmp_path
    )
    from my_jev.data import load_jsonl

    train_records = load_jsonl(
        paths["train"]
    )
    test_records = load_jsonl(
        paths["test"]
    )
    leaked_family = (
        train_records[0]
        .metadata["family_id"]
    )
    test_records[0].metadata[
        "family_id"
    ] = leaked_family
    dump_jsonl(
        iter(test_records),
        paths["test"],
    )

    with pytest.raises(
        ValueError,
        match="family_id leakage",
    ):
        audit_dataset_splits(
            paths
        )


def test_data_audit_rejects_unlabeled_records(
    tmp_path,
):
    paths = _write_split_set(
        tmp_path
    )
    from my_jev.data import load_jsonl

    records = load_jsonl(
        paths["validation"]
    )
    records[0].targets = None
    dump_jsonl(
        iter(records),
        paths["validation"],
    )

    with pytest.raises(
        ValueError,
        match="unlabeled",
    ):
        audit_dataset_splits(
            paths
        )
