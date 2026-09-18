from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    actual: float
    operator: str
    threshold: float

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "actual": self.actual,
            "operator": self.operator,
            "threshold": self.threshold,
        }


def _metric(
    payload: dict[str, Any],
    path: tuple[str, ...],
) -> float:
    current: Any = payload
    for key in path:
        if not isinstance(
            current,
            dict,
        ):
            raise KeyError(
                ".".join(path)
            )
        current = current[
            key
        ]
    return float(current)


def evaluate_promotion(
    benchmark: dict[str, Any],
    gates: dict[str, float],
) -> dict[str, object]:
    """Evaluate explicit promotion gates against one benchmark artifact."""
    definitions: dict[
        str,
        tuple[
            tuple[str, ...],
            str,
        ],
    ] = {
        "min_accuracy": (
            ("normal", "accuracy"),
            ">=",
        ),
        "max_ece": (
            ("normal", "ece"),
            "<=",
        ),
        "max_nll": (
            ("normal", "nll"),
            "<=",
        ),
        "max_brier": (
            ("normal", "brier"),
            "<=",
        ),
        "min_accuracy_delta_vs_shuffled": (
            (
                "controls",
                "accuracy_delta_vs_shuffled",
            ),
            ">=",
        ),
        "min_accuracy_delta_vs_uniform": (
            (
                "controls",
                "accuracy_delta_vs_uniform",
            ),
            ">=",
        ),
        "min_choice_order_top1_agreement": (
            (
                "controls",
                "choice_order_invariance",
                "top1_agreement",
            ),
            ">=",
        ),
        "max_choice_order_mean_abs_delta": (
            (
                "controls",
                "choice_order_invariance",
                "mean_abs_probability_delta",
            ),
            "<=",
        ),
        "max_policy_consistency_violation_rate": (
            (
                "controls",
                "policy_consistency",
                "violation_rate",
            ),
            "<=",
        ),
        "min_decisions_per_second": (
            (
                "latency",
                "decisions_per_second",
            ),
            ">=",
        ),
        "max_batch_ms_p95": (
            (
                "latency",
                "batch_ms_p95",
            ),
            "<=",
        ),
    }

    results: list[
        GateResult
    ] = []
    unknown = (
        set(gates)
        - set(definitions)
    )
    if unknown:
        raise ValueError(
            "unknown promotion gates: "
            f"{sorted(unknown)}"
        )

    for name, threshold in (
        gates.items()
    ):
        path, operator = (
            definitions[name]
        )
        actual = _metric(
            benchmark,
            path,
        )
        passed = (
            actual >= threshold
            if operator == ">="
            else actual <= threshold
        )
        results.append(
            GateResult(
                name=name,
                passed=passed,
                actual=actual,
                operator=operator,
                threshold=float(
                    threshold
                ),
            )
        )

    return {
        "passed": all(
            result.passed
            for result in results
        ),
        "gates": [
            result.as_dict()
            for result in results
        ],
        "failed": [
            result.name
            for result in results
            if not result.passed
        ],
    }
