from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from .schema import DecisionRecord, QuestionType, TargetSpec


class QuestionOutputLike(Protocol):
    record_index: int
    name: str
    type: QuestionType
    logits: Tensor


@dataclass(frozen=True)
class LossWeights:
    nll: float = 1.0
    brier: float = 0.25
    ordinal: float = 0.20
    calibration: float = 0.05


def target_distribution(target: TargetSpec, size: int, device: torch.device) -> Tensor:
    if target.distribution is not None:
        return torch.tensor(target.distribution, dtype=torch.float32, device=device)
    distribution = torch.zeros(size, dtype=torch.float32, device=device)
    assert target.index is not None
    distribution[target.index] = 1.0
    return distribution


def brier_loss(probabilities: Tensor, target: Tensor) -> Tensor:
    return torch.sum((probabilities - target) ** 2)


def ordinal_emd_loss(probabilities: Tensor, target: Tensor) -> Tensor:
    """Squared earth-mover distance for ordered Score levels."""
    if probabilities.numel() <= 1:
        return probabilities.new_zeros(())
    predicted_cdf = torch.cumsum(probabilities, 0)[:-1]
    target_cdf = torch.cumsum(target, 0)[:-1]
    return torch.mean((predicted_cdf - target_cdf) ** 2)


def confidence_calibration_loss(probabilities: Tensor, target_index: int) -> Tensor:
    confidence, prediction = probabilities.max(dim=-1)
    correct = (prediction.detach() == target_index).to(probabilities.dtype)
    return (confidence - correct) ** 2


def decision_loss(
    output: QuestionOutputLike,
    target: TargetSpec,
    *,
    weights: LossWeights = LossWeights(),
) -> tuple[Tensor, dict[str, float]]:
    probabilities = torch.softmax(output.logits, dim=-1)
    target_dist = target_distribution(target, output.logits.numel(), output.logits.device)

    nll = -(target_dist * torch.log_softmax(output.logits, dim=-1)).sum()
    brier = brier_loss(probabilities, target_dist)
    ordinal = (
        ordinal_emd_loss(probabilities, target_dist)
        if output.type == QuestionType.SCORE
        else probabilities.new_zeros(())
    )
    hard_index = target.index if target.index is not None else int(target_dist.argmax().item())
    calibration = confidence_calibration_loss(probabilities, hard_index)

    total = (
        weights.nll * nll
        + weights.brier * brier
        + weights.ordinal * ordinal
        + weights.calibration * calibration
    )
    metrics = {
        "loss": float(total.detach().item()),
        "nll": float(nll.detach().item()),
        "brier": float(brier.detach().item()),
        "ordinal": float(ordinal.detach().item()),
        "calibration": float(calibration.detach().item()),
    }
    return total, metrics


def batch_loss(
    outputs: list[QuestionOutputLike],
    records: list[DecisionRecord],
    *,
    weights: LossWeights = LossWeights(),
) -> tuple[Tensor, dict[str, float]]:
    losses: list[Tensor] = []
    totals = {
        "loss": 0.0,
        "nll": 0.0,
        "brier": 0.0,
        "ordinal": 0.0,
        "calibration": 0.0,
    }

    for output in outputs:
        targets = records[output.record_index].targets
        if not targets or output.name not in targets:
            continue
        loss, metrics = decision_loss(output, targets[output.name], weights=weights)
        losses.append(loss)
        for key in totals:
            totals[key] += metrics[key]

    if not losses:
        raise ValueError("batch contains no labeled questions")

    count = float(len(losses))
    return torch.stack(losses).mean(), {
        key: value / count for key, value in totals.items()
    }
