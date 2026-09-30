from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .registry import (
    ExperimentEntry,
    ExperimentRegistry,
)


_METRICS: tuple[
    tuple[
        str,
        tuple[str, ...],
    ],
    ...,
] = (
    (
        "accuracy",
        (
            "normal",
            "accuracy",
        ),
    ),
    (
        "ece",
        (
            "normal",
            "ece",
        ),
    ),
    (
        "nll",
        (
            "normal",
            "nll",
        ),
    ),
    (
        "brier",
        (
            "normal",
            "brier",
        ),
    ),
    (
        "state_delta",
        (
            "controls",
            "accuracy_delta_vs_shuffled",
        ),
    ),
    (
        "uniform_delta",
        (
            "controls",
            "accuracy_delta_vs_uniform",
        ),
    ),
    (
        "choice_top1_agreement",
        (
            "controls",
            "choice_order_invariance",
            "top1_agreement",
        ),
    ),
    (
        "choice_mean_abs_delta",
        (
            "controls",
            "choice_order_invariance",
            "mean_abs_probability_delta",
        ),
    ),
    (
        "policy_violation_rate",
        (
            "controls",
            "policy_consistency",
            "violation_rate",
        ),
    ),
    (
        "decisions_per_second",
        (
            "latency",
            "decisions_per_second",
        ),
    ),
    (
        "batch_ms_p95",
        (
            "latency",
            "batch_ms_p95",
        ),
    ),
)


def _get(
    payload: dict[str, Any],
    path: tuple[str, ...],
) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(
            current,
            dict,
        ):
            return None
        current = current.get(
            key
        )
        if current is None:
            return None
    return current


def _load_benchmark(
    entry: ExperimentEntry,
) -> dict[str, Any]:
    if not entry.benchmark_path:
        raise ValueError(
            f"run {entry.run_id} "
            "has no benchmark artifact"
        )
    path = Path(
        entry.benchmark_path
    )
    if not path.exists():
        raise FileNotFoundError(
            path
        )
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            f"benchmark for "
            f"{entry.run_id} "
            "is not a JSON object"
        )
    return payload


def summarize_run(
    entry: ExperimentEntry,
) -> dict[str, object]:
    benchmark = _load_benchmark(
        entry
    )
    metrics = {
        name: _get(
            benchmark,
            path,
        )
        for name, path in _METRICS
    }
    return {
        "run_id": entry.run_id,
        "experiment": (
            entry.experiment
        ),
        "status": entry.status,
        "promoted": entry.promoted,
        "backend": (
            entry.model_backend
        ),
        "backbone": entry.backbone,
        "spec_sha256": (
            entry.spec_sha256
        ),
        "datasets": {
            "train": (
                entry.train_sha256
            ),
            "validation": (
                entry.validation_sha256
            ),
            "calibration": (
                entry.calibration_sha256
            ),
            "test": (
                entry.test_sha256
            ),
        },
        "metrics": metrics,
    }


def compare_runs(
    registry: ExperimentRegistry,
    run_ids: list[str],
) -> dict[str, object]:
    if len(run_ids) < 2:
        raise ValueError(
            "comparison needs at least "
            "two run IDs"
        )
    entries = []
    for run_id in run_ids:
        entry = registry.get(
            run_id
        )
        if entry is None:
            raise KeyError(
                f"unknown run: {run_id}"
            )
        entries.append(
            entry
        )

    summaries = [
        summarize_run(
            entry
        )
        for entry in entries
    ]
    test_hashes = {
        summary[
            "datasets"
        ]["test"]
        for summary in summaries
    }
    calibration_hashes = {
        summary[
            "datasets"
        ]["calibration"]
        for summary in summaries
    }
    return {
        "comparable": (
            len(test_hashes) == 1
            and len(
                calibration_hashes
            ) == 1
        ),
        "same_test_split": (
            len(test_hashes) == 1
        ),
        "same_calibration_split": (
            len(
                calibration_hashes
            )
            == 1
        ),
        "runs": summaries,
    }


def latest_run_ids(
    registry: ExperimentRegistry,
    experiments: list[str],
    *,
    promoted_only: bool = False,
) -> list[str]:
    run_ids: list[str] = []
    for experiment in experiments:
        entry = registry.latest(
            experiment,
            promoted_only=(
                promoted_only
            ),
        )
        if entry is None:
            raise KeyError(
                "no matching run for "
                f"{experiment}"
            )
        run_ids.append(
            entry.run_id
        )
    return run_ids


def _markdown(
    comparison: dict[str, object],
) -> str:
    runs = comparison[
        "runs"
    ]
    assert isinstance(
        runs,
        list,
    )
    headers = [
        "metric",
        *[
            str(
                run["experiment"]
            )
            for run in runs
        ],
    ]
    rows = [
        headers,
        [
            "run",
            *[
                str(
                    run["run_id"]
                )
                for run in runs
            ],
        ],
        [
            "backend",
            *[
                str(
                    run["backend"]
                )
                for run in runs
            ],
        ],
        [
            "promoted",
            *[
                str(
                    run["promoted"]
                )
                for run in runs
            ],
        ],
    ]
    for name, _ in _METRICS:
        rows.append(
            [
                name,
                *[
                    str(
                        run[
                            "metrics"
                        ].get(
                            name
                        )
                    )
                    for run in runs
                ],
            ]
        )

    widths = [
        max(
            len(
                row[index]
            )
            for row in rows
        )
        for index in range(
            len(headers)
        )
    ]

    def render(
        row: list[str],
    ) -> str:
        return (
            "| "
            + " | ".join(
                value.ljust(
                    widths[index]
                )
                for index, value
                in enumerate(row)
            )
            + " |"
        )

    separator = (
        "| "
        + " | ".join(
            "-" * width
            for width in widths
        )
        + " |"
    )
    return "\n".join(
        [
            render(
                rows[0]
            ),
            separator,
            *[
                render(row)
                for row
                in rows[1:]
            ],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare registered my-jev "
            "experiment benchmarks"
        )
    )
    parser.add_argument(
        "--registry",
        default=(
            "runs/experiments/"
            "registry.json"
        ),
    )
    selector = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )
    selector.add_argument(
        "--runs",
        help=(
            "comma-separated run IDs"
        ),
    )
    selector.add_argument(
        "--latest",
        help=(
            "comma-separated experiment "
            "names; compare latest runs"
        ),
    )
    parser.add_argument(
        "--promoted-only",
        action="store_true",
    )
    parser.add_argument(
        "--format",
        choices=(
            "json",
            "markdown",
        ),
        default="markdown",
    )
    parser.add_argument(
        "--output",
    )
    args = parser.parse_args()

    registry = ExperimentRegistry(
        args.registry
    )
    if args.runs:
        run_ids = [
            value.strip()
            for value
            in args.runs.split(",")
            if value.strip()
        ]
    else:
        experiments = [
            value.strip()
            for value
            in args.latest.split(",")
            if value.strip()
        ]
        run_ids = latest_run_ids(
            registry,
            experiments,
            promoted_only=(
                args.promoted_only
            ),
        )

    comparison = compare_runs(
        registry,
        run_ids,
    )
    output = (
        json.dumps(
            comparison,
            indent=2,
            sort_keys=True,
        )
        if args.format == "json"
        else _markdown(
            comparison
        )
    )
    print(output)
    if args.output:
        destination = Path(
            args.output
        )
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        destination.write_text(
            output + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
