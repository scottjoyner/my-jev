from __future__ import annotations

import pytest

from my_jev.assistx_policy_eval import evaluate_policy_predictions
from my_jev.schema import DecisionRecord


def _record():
    return DecisionRecord.model_validate(
        {
            "state": "request",
            "questions": {
                "execution_policy": {
                    "type": "choice",
                    "instructions": "Choose policy",
                    "options": ["exec-a", "exec-b", "exec-c"],
                }
            },
            "targets": {
                "execution_policy": {
                    "distribution": [0.5, 0.5, 0.0],
                }
            },
            "metadata": {
                "domain": "assistx_inference_policy",
                "family_id": "case-a",
                "assistx_record_id": "r1",
                "context_tokens": 32768,
                "best_wall_ms": 100.0,
                "near_best_signature_ids": ["exec-a", "exec-b"],
                "candidate_evidence": {
                    "exec-a": {"wall_ms": 100.0},
                    "exec-b": {"wall_ms": 102.0},
                    "exec-c": {"wall_ms": 150.0},
                },
            },
        }
    )


def test_heldout_eval_reports_counterfactual_regret_without_dispatch():
    record = _record()
    report = evaluate_policy_predictions(
        [record],
        [
            {
                "record_index": 0,
                "question": "execution_policy",
                "options": ["exec-a", "exec-b", "exec-c"],
                "probabilities": [0.1, 0.8, 0.1],
                "choice": "exec-b",
                "confidence": 0.8,
            }
        ],
    )

    assert report["near_best_hit_rate"] == 1.0
    assert report["aggregate_regret_ms"] == 2.0
    assert report["aggregate_regret_ratio"] == 1.02
    assert report["records"][0]["expected_wall_ms"] == pytest.approx(106.6)
    assert report["authority"]["dispatch_allowed"] is False


def test_slower_choice_records_positive_regret():
    report = evaluate_policy_predictions(
        [_record()],
        [
            {
                "record_index": 0,
                "question": "execution_policy",
                "options": ["exec-a", "exec-b", "exec-c"],
                "probabilities": [0.1, 0.1, 0.8],
                "choice": "exec-c",
                "confidence": 0.8,
            }
        ],
    )

    assert report["near_best_hit_rate"] == 0.0
    assert report["mean_regret_ms"] == 50.0
    assert report["mean_regret_ratio"] == 1.5


def test_prediction_option_drift_is_rejected():
    with pytest.raises(ValueError, match="option ordering drift"):
        evaluate_policy_predictions(
            [_record()],
            [
                {
                    "record_index": 0,
                    "question": "execution_policy",
                    "options": ["exec-b", "exec-a", "exec-c"],
                    "probabilities": [0.8, 0.1, 0.1],
                    "choice": "exec-b",
                }
            ],
        )


def test_prediction_probability_drift_is_rejected():
    with pytest.raises(ValueError, match="sum to one"):
        evaluate_policy_predictions(
            [_record()],
            [
                {
                    "record_index": 0,
                    "question": "execution_policy",
                    "options": ["exec-a", "exec-b", "exec-c"],
                    "probabilities": [0.2, 0.2, 0.2],
                    "choice": "exec-a",
                }
            ],
        )
