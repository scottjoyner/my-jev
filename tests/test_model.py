import pytest
import torch

from my_jev.model import (
    DynamicDecisionHead,
    QuestionOutput,
)
from my_jev.schema import QuestionType


def test_dynamic_decision_head_scores_candidates_and_backprops():
    torch.manual_seed(7)
    head = DynamicDecisionHead(
        hidden_size=8,
        num_heads=2,
        dropout=0.0,
    )
    state_hidden = torch.randn(
        2,
        4,
        8,
        requires_grad=True,
    )
    state_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ]
    )
    candidate_pooled = torch.randn(
        3,
        8,
        requires_grad=True,
    )
    candidate_state_index = torch.tensor(
        [0, 1, 1],
        dtype=torch.long,
    )

    logits = head(
        state_hidden=state_hidden,
        state_mask=state_mask,
        candidate_pooled=candidate_pooled,
        candidate_state_index=candidate_state_index,
    )

    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()

    logits.sum().backward()
    assert candidate_pooled.grad is not None
    assert torch.isfinite(
        candidate_pooled.grad
    ).all()
    assert state_hidden.grad is not None


def test_question_output_temperature_changes_confidence():
    output = QuestionOutput(
        record_index=0,
        name="route",
        type=QuestionType.CHOICE,
        options=["app", "database"],
        logits=torch.tensor([2.0, 0.0]),
    )

    raw = output.probabilities
    softened = output.probabilities_at_temperature(
        2.0
    )

    assert float(raw.sum()) == pytest.approx(1.0)
    assert float(softened.sum()) == pytest.approx(1.0)
    assert float(softened.max()) < float(raw.max())

    with pytest.raises(ValueError):
        output.probabilities_at_temperature(0.0)
