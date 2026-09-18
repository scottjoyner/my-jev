from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationMetrics:
    accuracy: float
    nll: float
    brier: float
    ece: float
    count: int


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    bins: int = 15,
) -> float:
    if confidences.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        left, right = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)
        if not np.any(mask):
            continue
        weight = float(mask.mean())
        ece += weight * abs(
            float(correct[mask].mean()) - float(confidences[mask].mean())
        )
    return float(ece)


def multiclass_metrics(
    probabilities: list[np.ndarray],
    targets: list[int],
) -> CalibrationMetrics:
    if not probabilities:
        return CalibrationMetrics(0.0, 0.0, 0.0, 0.0, 0)

    confidences = []
    correctness = []
    nll_values = []
    brier_values = []

    for probs, target in zip(probabilities, targets, strict=True):
        probs = np.asarray(probs, dtype=np.float64)
        probs = np.clip(probs, 0.0, None)
        total = probs.sum()
        if total <= 0:
            raise ValueError("probability vector has no positive mass")
        probs /= total
        pred = int(probs.argmax())
        confidences.append(float(probs[pred]))
        correctness.append(float(pred == target))
        nll_values.append(float(-np.log(max(probs[target], 1e-12))))
        one_hot = np.zeros_like(probs)
        one_hot[target] = 1.0
        brier_values.append(float(np.square(probs - one_hot).sum()))

    conf = np.asarray(confidences)
    corr = np.asarray(correctness)
    return CalibrationMetrics(
        accuracy=float(corr.mean()),
        nll=float(np.mean(nll_values)),
        brier=float(np.mean(brier_values)),
        ece=expected_calibration_error(conf, corr),
        count=len(targets),
    )
