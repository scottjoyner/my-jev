import torch

from my_jev.activation_protocol import (
    ActivationFrame,
    ActivationQuestionGroup,
)
from my_jev.external_decision import (
    ExternalActivationDecisionModel,
)


def _frame() -> ActivationFrame:
    return ActivationFrame(
        provider=(
            "bonsai2-ternary-llama-cpp"
        ),
        model_sha256="a" * 64,
        runtime_revision="b" * 40,
        prompt_contract=(
            "assistx-agent-policy-v1"
        ),
        state_taps=torch.randn(
            1,
            4,
            6,
            16,
        ),
        state_mask=torch.tensor(
            [
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                ]
            ],
            dtype=torch.bool,
        ),
        option_taps=torch.randn(
            1,
            4,
            4,
            16,
        ),
        option_mask=torch.tensor(
            [
                [
                    1,
                    1,
                    1,
                    1,
                ]
            ],
            dtype=torch.bool,
        ),
        groups=(
            ActivationQuestionGroup(
                record_index=0,
                name="route",
                type="choice",
                options=(
                    "chat",
                    "act",
                ),
                option_start=0,
                option_end=2,
            ),
            ActivationQuestionGroup(
                record_index=0,
                name="needs_tools",
                type="noul",
                options=(
                    "false",
                    "true",
                ),
                option_start=2,
                option_end=4,
            ),
        ),
    )


def test_external_model_returns_standard_question_outputs():
    torch.manual_seed(
        107
    )
    model = (
        ExternalActivationDecisionModel(
            provider=(
                "bonsai2-ternary-llama-cpp"
            ),
            hidden_size=16,
            tap_count=4,
            rank=8,
            dropout=0.0,
        )
    )

    outputs = (
        model.forward_frame(
            _frame()
        )
    )

    assert (
        len(
            outputs
        )
        == 2
    )
    assert (
        outputs[
            0
        ].name
        == "route"
    )
    assert (
        outputs[
            0
        ].logits.shape
        == (
            2,
        )
    )
    assert (
        outputs[
            1
        ].name
        == "needs_tools"
    )


def test_external_model_predicts_same_typed_contract_shape():
    torch.manual_seed(
        109
    )
    model = (
        ExternalActivationDecisionModel(
            provider=(
                "bonsai2-ternary-llama-cpp"
            ),
            hidden_size=16,
            tap_count=4,
            rank=8,
            dropout=0.0,
        )
    )

    predictions = (
        model.predict_frame(
            _frame(),
            temperature=1.2,
        )
    )

    assert (
        predictions[
            0
        ]["question"]
        == "route"
    )
    assert (
        predictions[
            1
        ]["question"]
        == "needs_tools"
    )
    assert (
        predictions[
            1
        ]["noul"]
        >= 0.0
    )
    assert (
        predictions[
            0
        ]["provider"]
        == (
            "bonsai2-ternary-llama-cpp"
        )
    )
