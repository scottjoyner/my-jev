import pytest

from my_jev.schema import DecisionRecord
from my_jev.teacher import (
    aggregate_teacher_targets,
    parse_teacher_targets,
)


def record():
    return DecisionRecord.model_validate(
        {
            "state": "checkout is failing",
            "questions": {
                "urgent": {
                    "type": "noul",
                    "instructions": "urgent?",
                },
                "route": {
                    "type": "choice",
                    "instructions": "route?",
                    "options": ["app", "db", "network"],
                },
            },
        }
    )


def test_parse_teacher_targets_normalizes():
    targets = parse_teacher_targets(
        '{"targets":{"urgent":{"distribution":[1,3]},'
        '"route":{"distribution":[2,1,1]}}}',
        record(),
    )
    assert targets["urgent"].distribution == [0.25, 0.75]
    assert targets["route"].distribution == [0.5, 0.25, 0.25]


def test_aggregate_teacher_targets_averages_distributions():
    item = record()
    first = parse_teacher_targets(
        '{"targets":{"urgent":{"distribution":[0.2,0.8]},'
        '"route":{"distribution":[1,0,0]}}}',
        item,
    )
    second = parse_teacher_targets(
        '{"targets":{"urgent":{"distribution":[0.6,0.4]},'
        '"route":{"distribution":[0,1,0]}}}',
        item,
    )
    averaged = aggregate_teacher_targets(
        item,
        [first, second],
    )
    assert averaged["urgent"].distribution == pytest.approx(
        [0.4, 0.6]
    )
    assert averaged["route"].distribution == [
        0.5,
        0.5,
        0.0,
    ]
