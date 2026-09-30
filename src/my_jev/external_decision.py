from __future__ import annotations

import torch
from torch import nn

from .activation_protocol import (
    ActivationFrame,
)
from .decision_attention import (
    ExternalDecisionAttentionHead,
)
from .model import QuestionOutput
from .schema import QuestionType


class ExternalActivationDecisionModel(
    nn.Module
):
    """Typed decision model over activations supplied by an external runtime."""

    head_kind = (
        "external_decision_attention"
    )

    def __init__(
        self,
        *,
        provider: str,
        hidden_size: int,
        tap_count: int,
        rank: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not provider:
            raise ValueError(
                "provider must be non-empty"
            )

        self.provider = provider
        self.backbone_name = (
            provider
        )
        self.head_rank = rank
        self.head = (
            ExternalDecisionAttentionHead(
                hidden_size=(
                    hidden_size
                ),
                tap_count=(
                    tap_count
                ),
                rank=rank,
                dropout=dropout,
            )
        )

    def forward_frame(
        self,
        frame: ActivationFrame,
        *,
        shuffle_state: bool = False,
    ) -> list[
        QuestionOutput
    ]:
        frame.validate()
        if (
            frame.provider
            != self.provider
        ):
            raise ValueError(
                "activation provider "
                f"{frame.provider!r} does not "
                "match model provider "
                f"{self.provider!r}"
            )

        device = next(
            self.parameters()
        ).device
        logits = self.head(
            frame.state_taps.to(
                device
            ),
            frame.state_mask.to(
                device
            ),
            frame.option_taps.to(
                device
            ),
            frame.option_mask.to(
                device
            ),
            shuffle_state=(
                shuffle_state
            ),
        )

        outputs: list[
            QuestionOutput
        ] = []
        for group in (
            frame.groups
        ):
            try:
                question_type = (
                    QuestionType(
                        group.type
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    "activation frame has "
                    "unknown question type "
                    f"{group.type!r}"
                ) from exc

            outputs.append(
                QuestionOutput(
                    record_index=(
                        group.record_index
                    ),
                    name=group.name,
                    type=question_type,
                    options=list(
                        group.options
                    ),
                    logits=logits[
                        group.record_index,
                        group.option_start
                        : group.option_end,
                    ],
                )
            )

        return outputs

    def predict_frame(
        self,
        frame: ActivationFrame,
        *,
        temperature: float = 1.0,
    ) -> list[
        dict[str, object]
    ]:
        if temperature <= 0:
            raise ValueError(
                "temperature must be positive"
            )

        self.eval()
        with torch.inference_mode():
            outputs = self.forward_frame(
                frame
            )

        predictions: list[
            dict[str, object]
        ] = []
        for output in outputs:
            probabilities = (
                output
                .probabilities_at_temperature(
                    temperature
                )
                .detach()
                .cpu()
            )
            best = int(
                probabilities.argmax()
                .item()
            )
            payload: dict[
                str,
                object,
            ] = {
                "record_index": (
                    output.record_index
                ),
                "question": (
                    output.name
                ),
                "type": (
                    output.type.value
                ),
                "options": (
                    output.options
                ),
                "probabilities": (
                    probabilities.tolist()
                ),
                "choice": (
                    output.options[
                        best
                    ]
                ),
                "confidence": float(
                    probabilities[
                        best
                    ].item()
                ),
                "temperature": (
                    temperature
                ),
                "provider": (
                    self.provider
                ),
            }

            if (
                output.type
                == QuestionType.NOUL
            ):
                payload["noul"] = (
                    float(
                        probabilities[
                            1
                        ].item()
                    )
                )
            elif (
                output.type
                == QuestionType.SCORE
            ):
                positions = (
                    torch.linspace(
                        0.0,
                        1.0,
                        len(
                            output.options
                        ),
                    )
                )
                payload["score"] = (
                    float(
                        (
                            positions
                            * probabilities
                        ).sum()
                        .item()
                    )
                )

            predictions.append(
                payload
            )

        return predictions
