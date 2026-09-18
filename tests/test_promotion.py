import pytest

from my_jev.promotion import evaluate_promotion


def _benchmark():
    return {
        "normal": {
            "accuracy": 0.91,
            "ece": 0.04,
            "nll": 0.22,
            "brier": 0.08,
        },
        "controls": {
            "accuracy_delta_vs_shuffled": 0.31,
            "accuracy_delta_vs_uniform": 0.42,
            "choice_order_invariance": {
                "top1_agreement": 0.995,
                "mean_abs_probability_delta": 0.003,
            },
            "policy_consistency": {
                "violation_rate": 0.01,
            },
        },
        "latency": {
            "decisions_per_second": 850.0,
            "batch_ms_p95": 42.0,
        },
    }


def test_promotion_passes_explicit_behavioral_gates():
    result = evaluate_promotion(
        _benchmark(),
        {
            "min_accuracy": 0.85,
            "max_ece": 0.08,
            "min_accuracy_delta_vs_shuffled": 0.15,
            "min_choice_order_top1_agreement": 0.98,
            "max_choice_order_mean_abs_delta": 0.02,
            "max_policy_consistency_violation_rate": 0.02,
        },
    )
    assert result["passed"] is True
    assert result["failed"] == []


def test_promotion_rejects_consistency_regression():
    benchmark = _benchmark()
    benchmark["controls"]["policy_consistency"]["violation_rate"] = 0.12

    result = evaluate_promotion(
        benchmark,
        {
            "max_policy_consistency_violation_rate": 0.02,
        },
    )
    assert result["passed"] is False
    assert result["failed"] == [
        "max_policy_consistency_violation_rate"
    ]


def test_unknown_gate_is_rejected():
    with pytest.raises(
        ValueError,
        match="unknown promotion gates",
    ):
        evaluate_promotion(
            _benchmark(),
            {"magic_score": 1.0},
        )
