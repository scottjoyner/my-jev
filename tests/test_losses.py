from types import SimpleNamespace

import pytest
import torch

from my_jev.losses import (
    batch_loss,
    brier_loss,
    decision_loss,
    ordinal_emd_loss,
)
from my_jev.schema import DecisionRecord, QuestionType, TargetSpec


def test_brier_zero_for_exact_target():
    probabilities = torch.tensor([0.0, 1.0])
    target = torch.tensor([0.0, 1.0])
    assert float(
        brier_loss(
            probabilities,
            target,
        )
    ) == 0.0


def test_ordinal_near_miss_costs_less_than_far_miss():
    target = torch.tensor(
        [0.0, 0.0, 1.0, 0.0]
    )
    near = torch.tensor(
        [0.0, 1.0, 0.0, 0.0]
    )
    far = torch.tensor(
        [1.0, 0.0, 0.0, 0.0]
    )
    assert ordinal_emd_loss(
        near,
        target,
    ) < ordinal_emd_loss(
        far,
        target,
    )


def test_batch_loss_respects_question_loss_weight():
    record = DecisionRecord.model_validate(
        {
            "state": "agent turn",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "route?",
                    "options": ["chat", "act"],
                    "metadata": {
                        "loss_weight": 3.0
                    },
                },
                "detail": {
                    "type": "choice",
                    "instructions": "detail?",
                    "options": ["a", "b"],
                },
            },
            "targets": {
                "route": {"index": 1},
                "detail": {"index": 0},
            },
        }
    )
    route = SimpleNamespace(
        record_index=0,
        name="route",
        type=QuestionType.CHOICE,
        logits=torch.tensor(
            [2.0, 0.0],
            requires_grad=True,
        ),
    )
    detail = SimpleNamespace(
        record_index=0,
        name="detail",
        type=QuestionType.CHOICE,
        logits=torch.tensor(
            [2.0, 0.0],
            requires_grad=True,
        ),
    )

    route_loss, _ = decision_loss(
        route,
        TargetSpec(index=1),
    )
    detail_loss, _ = decision_loss(
        detail,
        TargetSpec(index=0),
    )
    combined, _ = batch_loss(
        [route, detail],
        [record],
    )

    expected = (
        3.0 * route_loss
        + detail_loss
    ) / 4.0
    assert float(
        combined.detach()
    ) == pytest.approx(
        float(expected.detach())
    )
    assert combined > (
        route_loss + detail_loss
    ) / 2
