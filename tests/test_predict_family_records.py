"""The adapter that turns model output into the predictions the evaluator reads.

`_predict_family_records` is on the path to every fleet-family metric: it converts
`QuestionOutput` objects into the `list[dict[question, dict[option, probability]]]`
that `evaluate_fleet_families` consumes, and it was the one function in that path
with no test at all.

It had no test because a *dead* near-duplicate of it sat beside it looking like the
real thing. That duplicate, `prediction_from_outputs`, has been removed; see
`docs/ARCHITECTURE.md`.
"""

from __future__ import annotations

import pytest

from my_jev.fleet_benchmark import _predict_family_records
from my_jev.fleet_family_eval import evaluate_fleet_families
from my_jev.fleet_synth import generate_fleet_records
from my_jev.model import QuestionOutput, QuestionType


class _StubModel:
    """Returns one `QuestionOutput` per question, with logits the test controls."""

    def __init__(self, per_record: int, *, temperature_offset: float = 0.0):
        self.per_record = per_record
        self.temperature_offset = temperature_offset
        self.batches: list[int] = []

    def forward_records(self, records):
        self.batches.append(len(records))
        outputs = []
        for index, record in enumerate(records):
            for position, (name, question) in enumerate(record.questions.items()):
                # Peaked on the first option so `_top1` has an unambiguous winner.
                options = question.options or []
                values = [(-3.0 + self.temperature_offset) if i == 0 else 0.0
                          for i in range(len(options))]
                outputs.append(
                    QuestionOutput(
                        record_index=index,
                        name=name,
                        type=QuestionType.CHOICE,
                        options=list(options),
                        logits=_logits(values),
                    )
                )
        return outputs


def _logits(values):
    """One question's logits, shaped (n_options,) exactly as the model slices them.

    `model.py` indexes the batch tensor by record and option range, so a
    question's logits are 1-D. A stub shaped (1, n) constructs fine and then
    fails several lines later inside `float()`, which is how this shape was
    pinned down rather than guessed.
    """
    import torch

    return torch.tensor(values, dtype=torch.float32)


def _family_records(count=6):
    return generate_fleet_records(count, seed=51)


def test_predictions_are_aligned_to_records_by_record_index():
    """The property that makes the output usable at all.

    Batching means one `forward_records` call covers several records, and each
    output carries a `record_index` local to its batch. The adapter has to add the
    batch offset; a prediction placed at its batch-local index instead would land on
    the wrong record and quietly corrupt every family metric derived from it.
    """
    records = _family_records(6)
    predictions = _predict_family_records(
        _StubModel(0), records, temperature=1.0, batch_size=2
    )

    assert len(predictions) == len(records)
    for index, prediction in enumerate(predictions):
        assert prediction, f"record {index} received no prediction"
        assert sorted(prediction) == sorted(records[index].questions)


def test_batching_does_not_change_the_result():
    """One batch of six and three batches of two must agree exactly.

    If they diverge, some result depends on the batch size, which is a property of
    the caller's flag rather than of the model.
    """
    records = _family_records(6)
    single = _predict_family_records(_StubModel(0), records, temperature=1.0, batch_size=6)
    split = _predict_family_records(_StubModel(0), records, temperature=1.0, batch_size=2)
    assert single == split


def test_temperature_actually_changes_the_distribution():
    """The transform is the reason this function exists.

    A duplicate of this adapter sat unused beside it, and that duplicate read raw
    probabilities. Wiring it up would have fed untempered numbers into every family
    metric, quietly.
    """
    records = _family_records(2)
    hot = _predict_family_records(_StubModel(0), records, temperature=0.5, batch_size=2)
    cool = _predict_family_records(_StubModel(0), records, temperature=8.0, batch_size=2)

    def peaked(predictions):
        # One record's worth: the adapter returns a list aligned to records.
        prediction = predictions[0]
        question = next(iter(prediction))
        return max(prediction[question].values())

    assert peaked(hot) > peaked(cool)


def test_probabilities_are_a_distribution_per_question():
    records = _family_records(2)
    predictions = _predict_family_records(
        _StubModel(0), records, temperature=1.0, batch_size=2
    )
    for prediction in predictions:
        for name, question in zip(
            prediction, records[0].questions.values(), strict=False
        ):
            options = question.options or []
            assert sorted(prediction[name]) == sorted(options)
            total = sum(prediction[name].values())
            assert total == pytest.approx(1.0, abs=1e-5)


def test_a_batch_size_below_one_is_refused():
    records = _family_records(2)
    with pytest.raises(ValueError, match="batch size"):
        _predict_family_records(
            _StubModel(0), records, temperature=1.0, batch_size=0
        )


def test_non_positive_temperature_is_refused():
    """`probabilities_at_temperature` guards this; the guard is worth pinning."""
    records = _family_records(1)
    with pytest.raises(ValueError, match="temperature"):
        _predict_family_records(
            _StubModel(0), records, temperature=0.0, batch_size=1
        )


def test_the_output_the_evaluator_consumes_is_exactly_this_shape():
    """The contract between the adapter and the evaluator, asserted end to end.

    The evaluator indexes `predictions` by record position and looks up each
    question by name. A shape that is right per-record but wrong overall would
    either raise here or, worse, evaluate one record's numbers against another's.
    """
    records = _family_records(24)
    predictions = _predict_family_records(
        _StubModel(0), records, temperature=1.0, batch_size=4
    )
    metrics = evaluate_fleet_families(records, predictions)
    assert isinstance(metrics, dict)
    # A shape mismatch must be refused rather than silently misaligned.
    with pytest.raises(ValueError, match="length mismatch"):
        evaluate_fleet_families(records, predictions[:-1])
