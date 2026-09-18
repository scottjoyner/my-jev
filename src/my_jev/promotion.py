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


def evaluate_regression(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    gates: dict[str, float],
) -> dict[str, object]:
    """Compare a candidate against its previously promoted parent benchmark."""
    definitions: dict[
        str,
        tuple[
            tuple[str, ...],
            str,
        ],
    ] = {
        "max_accuracy_drop": (
            ("normal", "accuracy"),
            "drop",
        ),
        "max_ece_increase": (
            ("normal", "ece"),
            "increase",
        ),
        "max_nll_increase": (
            ("normal", "nll"),
            "increase",
        ),
        "max_brier_increase": (
            ("normal", "brier"),
            "increase",
        ),
        "max_policy_consistency_violation_rate_increase": (
            (
                "controls",
                "policy_consistency",
                "violation_rate",
            ),
            "increase",
        ),
        "max_batch_ms_p95_ratio": (
            (
                "latency",
                "batch_ms_p95",
            ),
            "ratio",
        ),
        "max_decisions_per_second_drop_fraction": (
            (
                "latency",
                "decisions_per_second",
            ),
            "drop_fraction",
        ),
    }

    unknown = (
        set(gates)
        - set(definitions)
    )
    if unknown:
        raise ValueError(
            "unknown regression gates: "
            f"{sorted(unknown)}"
        )

    results: list[
        dict[str, object]
    ] = []
    for name, threshold in gates.items():
        path, mode = definitions[name]
        candidate_value = _metric(
            candidate,
            path,
        )
        baseline_value = _metric(
            baseline,
            path,
        )

        if mode == "drop":
            actual = (
                baseline_value
                - candidate_value
            )
        elif mode == "increase":
            actual = (
                candidate_value
                - baseline_value
            )
        elif mode == "ratio":
            actual = (
                candidate_value
                / baseline_value
                if baseline_value > 0
                else (
                    1.0
                    if candidate_value <= 0
                    else float("inf")
                )
            )
        elif mode == "drop_fraction":
            actual = (
                (
                    baseline_value
                    - candidate_value
                )
                / baseline_value
                if baseline_value > 0
                else 0.0
            )
        else:
            raise AssertionError(
                f"unsupported regression mode: {mode}"
            )

        passed = actual <= threshold
        results.append(
            {
                "name": name,
                "passed": passed,
                "candidate": (
                    candidate_value
                ),
                "baseline": (
                    baseline_value
                ),
                "actual": actual,
                "operator": "<=",
                "threshold": float(
                    threshold
                ),
            }
        )

    return {
        "passed": all(
            bool(
                result["passed"]
            )
            for result in results
        ),
        "gates": results,
        "failed": [
            str(
                result["name"]
            )
            for result in results
            if not result["passed"]
        ],
    }
