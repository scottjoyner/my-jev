"""Hermetic regression tests for the regression ``drop_fraction`` gate (R32).

Loads ``promotion.py`` directly by file path so these tests need neither the
``my_jev`` package ``__init__`` (pydantic) nor torch/transformers. All fixtures
are in-memory dicts; no model, file, network, service or live state is used.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "my_jev"
    / "promotion.py"
)


def _load_promotion():
    name = "_my_jev_promotion_r32_under_test"
    spec = importlib.util.spec_from_file_location(name, _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    # dataclasses needs the declaring module registered in sys.modules.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


promo = _load_promotion()

GATE = "max_decisions_per_second_drop_fraction"


def _throughput(value):
    return {"latency": {"decisions_per_second": value}}


def _result(baseline_value, candidate_value, threshold=0.10):
    return promo.evaluate_regression(
        _throughput(candidate_value),
        _throughput(baseline_value),
        {GATE: threshold},
    )


# --- Fail-closed paths (the bug) -----------------------------------------

def test_baseline_zero_fails_closed():
    result = _result(0.0, 0.0)
    assert result["passed"] is False
    assert result["failed"] == [GATE]
    assert result["gates"][0]["actual"] == float("inf")


def test_baseline_zero_with_positive_candidate_fails_closed():
    result = _result(0.0, 500.0)
    assert result["passed"] is False
    assert result["failed"] == [GATE]


def test_baseline_negative_with_worse_candidate_fails_closed():
    result = _result(-5.0, -100.0)
    assert result["passed"] is False
    assert result["failed"] == [GATE]


def test_baseline_negative_with_better_candidate_still_fails_closed():
    # No valid denominator exists even if the candidate "looks" better.
    result = _result(-5.0, -1.0)
    assert result["passed"] is False
    assert result["failed"] == [GATE]


def test_baseline_nonpositive_with_large_threshold_still_fails_closed():
    # inf must fail even a generous threshold.
    result = _result(0.0, 0.0, threshold=1e9)
    assert result["passed"] is False


# --- Unchanged controls --------------------------------------------------

def test_positive_baseline_small_drop_still_passes():
    result = _result(100.0, 95.0)
    assert result["passed"] is True
    assert result["failed"] == []
    assert result["gates"][0]["actual"] == pytest.approx(0.05)


def test_positive_baseline_large_drop_still_fails():
    result = _result(100.0, 50.0, threshold=0.20)
    assert result["passed"] is False
    assert result["failed"] == [GATE]


def test_equality_boundary_still_passes():
    result = _result(100.0, 90.0, threshold=0.10)
    assert result["passed"] is True


def test_nan_candidate_fails_closed():
    result = _result(100.0, float("nan"))
    assert result["passed"] is False


def test_ratio_mode_unchanged_on_invalid_baseline():
    baseline = {"latency": {"batch_ms_p95": 0.0}}
    candidate = {"latency": {"batch_ms_p95": 500.0}}
    result = promo.evaluate_regression(
        candidate,
        baseline,
        {"max_batch_ms_p95_ratio": 1.0},
    )
    assert result["passed"] is False
    assert result["gates"][0]["actual"] == float("inf")


def test_drop_and_increase_modes_unchanged():
    baseline = {"normal": {"accuracy": 0.90, "ece": 0.04}}
    candidate = {"normal": {"accuracy": 0.85, "ece": 0.08}}
    result = promo.evaluate_regression(
        candidate,
        baseline,
        {"max_accuracy_drop": 0.10, "max_ece_increase": 0.03},
    )
    assert result["passed"] is False
    assert result["failed"] == ["max_ece_increase"]


def test_promotion_evaluate_unchanged():
    benchmark = {"normal": {"accuracy": 0.90}}
    assert promo.evaluate_promotion(benchmark, {"min_accuracy": 0.85})["passed"] is True
    assert promo.evaluate_promotion(benchmark, {"min_accuracy": 0.95})["passed"] is False


def test_unknown_regression_gate_still_rejected():
    with pytest.raises(ValueError, match="unknown regression gates"):
        promo.evaluate_regression({}, {}, {"magic_gate": 1.0})
