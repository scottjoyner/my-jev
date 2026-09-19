import pytest
import torch

from my_jev.model import (
    DynamicDecisionHead,
    OptionQueryDecisionHead,
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


def test_option_query_head_scores_padded_options_without_state_expansion():
    torch.manual_seed(11)
    head = OptionQueryDecisionHead(
        hidden_size=8,
        rank=4,
        dropout=0.0,
    )
    state_hidden = torch.randn(
        2,
        5,
        8,
        requires_grad=True,
    )
    state_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    candidates = torch.randn(
        2,
        4,
        8,
        requires_grad=True,
    )
    candidate_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )

    logits = head(
        state_hidden,
        state_mask,
        candidates,
        candidate_mask,
    )

    assert logits.shape == (2, 4)
    assert torch.isfinite(logits[candidate_mask]).all()
    assert logits[0, 2] < -1e20

    logits[candidate_mask].sum().backward()
    assert candidates.grad is not None
    assert state_hidden.grad is not None


def test_option_query_shuffled_state_control_changes_scores():
    torch.manual_seed(17)
    head = OptionQueryDecisionHead(
        hidden_size=8,
        rank=8,
        dropout=0.0,
    )
    state_hidden = torch.randn(3, 5, 8)
    state_mask = torch.ones(3, 5, dtype=torch.bool)
    candidates = torch.randn(3, 2, 8)
    candidate_mask = torch.ones(3, 2, dtype=torch.bool)

    normal = head(
        state_hidden,
        state_mask,
        candidates,
        candidate_mask,
    )
    shuffled = head(
        state_hidden,
        state_mask,
        candidates,
        candidate_mask,
        shuffle_state=True,
    )
    assert not torch.allclose(normal, shuffled)


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
