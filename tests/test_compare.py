import json

from my_jev.compare import (
    compare_runs,
    latest_run_ids,
)
from my_jev.registry import (
    ExperimentEntry,
    ExperimentRegistry,
)


def _benchmark(path, *, accuracy, speed):
    payload = {
        "normal": {
            "accuracy": accuracy,
            "ece": 0.05,
            "nll": 0.2,
            "brier": 0.1,
        },
        "controls": {
            "accuracy_delta_vs_shuffled": 0.3,
            "accuracy_delta_vs_uniform": 0.4,
            "choice_order_invariance": {
                "top1_agreement": 1.0,
                "mean_abs_probability_delta": 0.0,
            },
            "policy_consistency": {
                "violation_rate": 0.0,
            },
        },
        "latency": {
            "decisions_per_second": speed,
            "batch_ms_p95": 25.0,
        },
    }
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _entry(
    run_id,
    experiment,
    benchmark,
    *,
    created,
):
    return ExperimentEntry(
        run_id=run_id,
        experiment=experiment,
        created_at=created,
        updated_at=created,
        status="candidate",
        spec_path="spec.toml",
        spec_sha256="a" * 64,
        model_backend=experiment,
        backbone="backbone",
        train_sha256="1" * 64,
        validation_sha256="2" * 64,
        calibration_sha256="3" * 64,
        test_sha256="4" * 64,
        run_dir=f"runs/{run_id}",
        benchmark_path=str(benchmark),
        promoted=True,
    )


def test_compare_requires_identical_eval_splits_for_comparable(tmp_path):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _benchmark(
        first_path,
        accuracy=0.9,
        speed=1000.0,
    )
    _benchmark(
        second_path,
        accuracy=0.92,
        speed=120.0,
    )

    registry = ExperimentRegistry(
        tmp_path / "registry.json"
    )
    registry.register(
        _entry(
            "run-a",
            "encoder",
            first_path,
            created=1.0,
        )
    )
    registry.register(
        _entry(
            "run-b",
            "causal",
            second_path,
            created=2.0,
        )
    )

    result = compare_runs(
        registry,
        ["run-a", "run-b"],
    )
    assert result["comparable"] is True
    assert (
        result["runs"][0]["metrics"]["accuracy"]
        == 0.9
    )
    assert (
        result["runs"][1]["metrics"]["decisions_per_second"]
        == 120.0
    )


def test_latest_run_ids_selects_each_experiment_family(tmp_path):
    benchmark = tmp_path / "benchmark.json"
    _benchmark(
        benchmark,
        accuracy=0.9,
        speed=100.0,
    )
    registry = ExperimentRegistry(
        tmp_path / "registry.json"
    )
    registry.register(
        _entry(
            "encoder-old",
            "encoder",
            benchmark,
            created=1.0,
        )
    )
    registry.register(
        _entry(
            "encoder-new",
            "encoder",
            benchmark,
            created=2.0,
        )
    )
    registry.register(
        _entry(
            "causal-new",
            "causal",
            benchmark,
            created=3.0,
        )
    )

    assert latest_run_ids(
        registry,
        ["encoder", "causal"],
        promoted_only=True,
    ) == [
        "encoder-new",
        "causal-new",
    ]
