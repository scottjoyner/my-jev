"""`split_records` decides the train/validation/calibration/test partition.

Two tests existed. The properties left uncovered are the ones that fail silently:

* a **different seed must produce a different split**. If the seed were ignored the
  partition would be fixed forever, and every "resample the split" step upstream
  would quietly do nothing while reporting success.
* the **fractions are validated**, with three distinct error paths. A typo like
  `--validation-fraction 1.0` that silently produced a training set overlapping the
  validation set would poison every metric computed from it.
* **leakage is impossible**, which is the function's reason for existing.

Everything here is pure and seeded, so the whole file runs in milliseconds and needs
no model, no dataset and no network.
"""

from __future__ import annotations

import pytest

from my_jev.split import _effective_group_key, split_records
from my_jev.synth import generate_synthetic_records

SPLIT_NAMES = ("train", "validation", "calibration", "test")


def _assignment(splits) -> dict[str, str]:
    """family id -> split name, for comparing two splits."""
    return {
        item.metadata["family_id"]: name
        for name, items in splits.items()
        for item in items
    }


# --- the property the function exists for ---------------------------------


def test_no_group_appears_in_two_splits_across_many_seeds():
    """Leakage is the reason grouping exists, so it is checked exhaustively.

    One seed proves little: a shuffled order can put every family of a given size on
    one side by luck. Fifty seeds makes an accidental leak implausible.
    """
    records = generate_synthetic_records(60, seed=13)
    for seed in range(50):
        splits, _ = split_records(
            records,
            train_fraction=0.6,
            validation_fraction=0.1,
            calibration_fraction=0.1,
            seed=seed,
            group_key="auto",
        )
        families = {
            name: {item.metadata["family_id"] for item in items}
            for name, items in splits.items()
        }
        for index, left in enumerate(SPLIT_NAMES):
            for right in SPLIT_NAMES[index + 1 :]:
                assert families[left].isdisjoint(families[right]), (
                    f"seed {seed}: {left} and {right} share a family"
                )


def test_every_record_lands_in_exactly_one_split():
    """No record dropped, none duplicated -- otherwise the corpus silently shrinks."""
    records = generate_synthetic_records(60, seed=13)
    splits, _ = split_records(records, seed=5, group_key="auto")
    seen = [
        item.state
        for items in splits.values()
        for item in items
    ]
    assert len(seen) == len(records)
    assert sorted(seen) == sorted(item.state for item in records)


# --- the seed --------------------------------------------------------------


def test_the_same_seed_reproduces_the_split_exactly():
    records = generate_synthetic_records(60, seed=13)
    first, _ = split_records(records, seed=19, group_key="auto")
    second, _ = split_records(records, seed=19, group_key="auto")
    assert _assignment(first) == _assignment(second)


def test_a_different_seed_produces_a_different_split():
    """The property that makes the seed worth having.

    Asserted rather than assumed. A shuffle that is seeded but ineffective would pass
    every determinism test -- two runs with the same seed still agree -- and would be
    invisible everywhere else.
    """
    records = generate_synthetic_records(60, seed=13)
    baseline, _ = split_records(records, seed=19, group_key="auto")
    differences = 0
    for seed in (20, 21, 22, 23):
        other, _ = split_records(records, seed=seed, group_key="auto")
        if _assignment(other) != _assignment(baseline):
            differences += 1
    assert differences >= 3, (
        "changing the seed barely changed the partition, so the seed is doing "
        "almost nothing"
    )


def test_the_order_records_arrive_in_does_not_change_the_split():
    """Otherwise the same corpus re-read in a different order trains a different model."""
    records = generate_synthetic_records(40, seed=21)
    forward, _ = split_records(records, seed=8, group_key="auto")
    backward, _ = split_records(list(reversed(records)), seed=8, group_key="auto")
    assert _assignment(forward) == _assignment(backward)


# --- fraction validation ---------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"validation_fraction": -0.1}, "must be >= 0"),
        ({"calibration_fraction": -0.5}, "must be >= 0"),
        ({"train_fraction": -0.1}, "must be >= 0"),
        ({"train_fraction": 0.0}, "must be > 0"),
        ({"train_fraction": 0.9, "validation_fraction": 0.2}, "must be < 1"),
        (
            {"train_fraction": 0.5, "validation_fraction": 0.5, "calibration_fraction": 0.1},
            "must be < 1",
        ),
    ],
)
def test_impossible_fractions_are_refused(kwargs, message):
    """A split that overlaps is worse than one that fails.

    Fractions summing to 1.0 or more would put the same group in two splits, which is
    precisely the leakage the grouping exists to prevent -- and it would do so after
    the metrics were already computed.
    """
    records = generate_synthetic_records(20, seed=1)
    with pytest.raises(ValueError, match=message):
        split_records(records, seed=1, group_key="auto", **kwargs)


def test_fractions_that_leave_a_holdout_are_allowed():
    """The remainder is the test split, so summing to less than 1 is intended."""
    records = generate_synthetic_records(20, seed=1)
    splits, _ = split_records(
        records,
        train_fraction=0.6,
        validation_fraction=0.1,
        calibration_fraction=0.1,
        seed=1,
        group_key="auto",
    )
    assert set(splits) == set(SPLIT_NAMES)
    assert all(splits[name] for name in SPLIT_NAMES)


# --- the grouping key ------------------------------------------------------


def test_auto_resolves_to_the_first_metadata_field_present_on_every_record():
    records = generate_synthetic_records(20, seed=4)
    _, group_key = split_records(records, group_key="auto")
    assert group_key is not None
    assert all(group_key in item.metadata for item in records)


def test_none_disables_grouping_so_each_record_is_its_own_group():
    records = generate_synthetic_records(20, seed=4)
    splits, group_key = split_records(records, group_key="none")
    assert group_key is None
    # With grouping off, families do get split -- which is exactly why the default is
    # "auto" rather than "none".
    families = {
        name: {item.metadata["family_id"] for item in items}
        for name, items in splits.items()
    }
    assert any(
        families[left] & families[right]
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    )


def test_the_literal_string_none_and_python_none_agree():
    records = generate_synthetic_records(20, seed=4)
    by_string, key_string = split_records(records, group_key="none")
    by_none, key_none = split_records(records, group_key=None)
    assert key_string is key_none is None
    assert _assignment(by_string) == _assignment(by_none)


def test_an_explicit_key_that_some_record_lacks_is_refused():
    """A grouping key that only some records carry would split families.

    Silently falling back to per-record grouping here would put family members in
    different splits -- the exact leakage this function exists to prevent, introduced
    by a typo in a key name.
    """
    records = generate_synthetic_records(20, seed=4)
    with pytest.raises(ValueError):
        split_records(records, group_key="not_a_real_field")


def test_effective_group_key_is_none_for_both_spellings():
    records = generate_synthetic_records(6, seed=4)
    assert _effective_group_key(records, None) is None
    assert _effective_group_key(records, "none") is None


# --- degenerate corpora ----------------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 3])
def test_a_tiny_corpus_keeps_every_record(count):
    """Fewer groups than splits must not raise or drop anything.

    Slicing with `int(total * fraction)` truncates, so a one-group corpus puts
    everything in the holdout and leaves `train` empty. That is a real possibility,
    not a hypothetical -- and it is why `train.main` refuses an empty training set
    rather than this function pretending it cannot happen. Asserted here so the
    behaviour is pinned rather than discovered.
    """
    records = generate_synthetic_records(count, seed=4)
    splits, _ = split_records(
        records,
        train_fraction=0.6,
        validation_fraction=0.2,
        calibration_fraction=0.1,
        seed=1,
        group_key="auto",
    )
    assert sum(len(items) for items in splits.values()) == count
    assert splits["train"] == [] or count >= 2


def test_an_empty_corpus_yields_empty_splits():
    splits, _ = split_records([], group_key="auto")
    assert all(splits[name] == [] for name in SPLIT_NAMES)


def test_records_without_any_grouping_metadata_still_split():
    """`auto` with nothing to group on falls back rather than raising.

    Worth pinning because it is the difference between "no grouping available" and
    "refused", and only one of those lets a corpus through.
    """
    records = generate_synthetic_records(12, seed=4)
    for item in records:
        item.metadata.clear()
    splits, group_key = split_records(records, group_key="auto")
    assert group_key is None
    assert sum(len(items) for items in splits.values()) == len(records)

# --- the entry point refuses what the splitter can legitimately produce -----


def test_train_refuses_an_empty_training_corpus(monkeypatch, tmp_path):
    """A split too small to fill the train slice is a legitimate output.

    `split_records` truncates `int(total_groups * fraction)`, so a one-group corpus
    yields an empty `train` and puts everything in the holdout. That is not something
    the splitter should refuse -- it did what it was asked.

    What must not happen downstream is `train.main` writing a checkpoint from a run
    that trained on nothing: with no batches the epoch body never executes,
    `best_nll` stays infinity, and the run still exits 0. An untrained model reported
    as a trained one is the failure the rest of this project spends its effort
    refusing.
    """
    import sys

    import my_jev.train as train_module

    empty = tmp_path / "train.jsonl"
    empty.write_text("", encoding="utf-8")
    valid = tmp_path / "valid.jsonl"
    valid.write_text(
        generate_synthetic_records(2, seed=1)[0].model_dump_json() + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "my_jev.train",
            "--train", str(empty),
            "--valid", str(valid),
            "--output", str(tmp_path / "out.pt"),
            "--epochs", "1",
            "--batch-size", "1",
            "--device", "cpu",
        ],
    )
    with pytest.raises(SystemExit, match="no training records"):
        train_module.main()


def test_train_refuses_an_empty_validation_corpus(monkeypatch, tmp_path):
    """Every reported metric would be undefined, which is not the same as zero."""
    import sys

    import my_jev.train as train_module

    train = tmp_path / "train.jsonl"
    train.write_text(
        generate_synthetic_records(2, seed=1)[0].model_dump_json() + "\n",
        encoding="utf-8",
    )
    empty = tmp_path / "valid.jsonl"
    empty.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "my_jev.train",
            "--train", str(train),
            "--valid", str(empty),
            "--output", str(tmp_path / "out.pt"),
            "--epochs", "1",
            "--device", "cpu",
        ],
    )
    with pytest.raises(SystemExit, match="no validation records"):
        train_module.main()
