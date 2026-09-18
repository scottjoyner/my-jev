from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import torch
from torch.utils.data import DataLoader

from .calibration import load_temperature
from .checkpoint import load_checkpoint
from .data import DecisionDataset, collate_records
from .manifest import dataset_manifest
from .metrics import CalibrationMetrics, multiclass_metrics
from .perturb import (
    align_probabilities_by_option,
    reverse_choice_options,
)
from .schema import QuestionType


def _target_value(target) -> int | np.ndarray:
    if target.distribution is not None:
        return np.asarray(
            target.distribution,
            dtype=np.float64,
        )
    assert target.index is not None
    return target.index


def _metrics_dict(
    probabilities: list[np.ndarray],
    targets: list[int | np.ndarray],
) -> dict[str, float | int]:
    metrics: CalibrationMetrics = multiclass_metrics(
        probabilities,
        targets,
    )
    return metrics.__dict__


def _percentile(
    values: list[float],
    q: float,
) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return (
        ordered[low] * (1.0 - weight)
        + ordered[high] * weight
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _uniform_like(
    probabilities: np.ndarray,
) -> np.ndarray:
    return np.full(
        probabilities.shape,
        1.0 / probabilities.size,
        dtype=np.float64,
    )


def _kl(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    p = np.clip(
        np.asarray(left, dtype=np.float64),
        1e-12,
        1.0,
    )
    q = np.clip(
        np.asarray(right, dtype=np.float64),
        1e-12,
        1.0,
    )
    p /= p.sum()
    q /= q.sum()
    return float(
        np.sum(
            p * np.log(p / q)
        )
    )


def benchmark(
    checkpoint: str,
    data: str,
    *,
    calibration: str | None = None,
    batch_size: int = 8,
) -> dict[str, object]:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model = load_checkpoint(
        checkpoint,
        device=device,
    )
    temperature = load_temperature(
        calibration
    )
    dataset = DecisionDataset(data)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_records,
    )

    normal_probs: list[np.ndarray] = []
    shuffled_probs: list[np.ndarray] = []
    uniform_probs: list[np.ndarray] = []
    targets: list[int | np.ndarray] = []
    latencies_ms: list[float] = []
    decision_count = 0
    state_count = 0
    sensitivity: list[float] = []

    choice_order_abs_delta: list[float] = []
    choice_order_kl: list[float] = []
    choice_order_top1: list[float] = []

    grouped_probs: dict[
        tuple[str, str],
        list[np.ndarray],
    ] = defaultdict(list)
    grouped_targets: dict[
        tuple[str, str],
        list[int | np.ndarray],
    ] = defaultdict(list)

    model.eval()
    with torch.inference_mode():
        for records in loader:
            _sync(device)
            started = time.perf_counter()
            outputs = model.forward_records(
                records
            )
            _sync(device)
            elapsed_ms = (
                time.perf_counter()
                - started
            ) * 1000.0
            latencies_ms.append(
                elapsed_ms
            )
            state_count += len(records)

            shuffled = model.forward_records(
                records,
                shuffle_state=True,
            )
            shuffled_by_key = {
                (
                    item.record_index,
                    item.name,
                ): item
                for item in shuffled
            }

            choice_reversed_records = [
                reverse_choice_options(
                    record
                )
                for record in records
            ]
            choice_reversed = (
                model.forward_records(
                    choice_reversed_records
                )
            )
            choice_reversed_by_key = {
                (
                    item.record_index,
                    item.name,
                ): item
                for item in choice_reversed
            }

            for output in outputs:
                record = records[
                    output.record_index
                ]
                target = (
                    record.targets or {}
                ).get(output.name)
                if target is None:
                    continue

                probability = (
                    output
                    .probabilities_at_temperature(
                        temperature
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                shuffled_output = (
                    shuffled_by_key[
                        (
                            output.record_index,
                            output.name,
                        )
                    ]
                )
                shuffled_probability = (
                    shuffled_output
                    .probabilities_at_temperature(
                        temperature
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                target_value = (
                    _target_value(
                        target
                    )
                )

                normal_probs.append(
                    probability
                )
                shuffled_probs.append(
                    shuffled_probability
                )
                uniform_probs.append(
                    _uniform_like(
                        probability
                    )
                )
                targets.append(
                    target_value
                )
                sensitivity.append(
                    _kl(
                        probability,
                        shuffled_probability,
                    )
                )
                decision_count += 1

                if (
                    output.type
                    == QuestionType.CHOICE
                ):
                    perturbed_output = (
                        choice_reversed_by_key[
                            (
                                output.record_index,
                                output.name,
                            )
                        ]
                    )
                    perturbed_probability = (
                        perturbed_output
                        .probabilities_at_temperature(
                            temperature
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    aligned = np.asarray(
                        align_probabilities_by_option(
                            reference_options=(
                                output.options
                            ),
                            candidate_options=(
                                perturbed_output.options
                            ),
                            candidate_probabilities=(
                                perturbed_probability.tolist()
                            ),
                        ),
                        dtype=np.float64,
                    )
                    absolute = np.abs(
                        probability - aligned
                    )
                    choice_order_abs_delta.extend(
                        absolute.tolist()
                    )
                    choice_order_kl.append(
                        _kl(
                            probability,
                            aligned,
                        )
                    )
                    choice_order_top1.append(
                        float(
                            int(
                                probability.argmax()
                                == aligned.argmax()
                            )
                        )
                    )

                groups = {
                    (
                        "question",
                        output.name,
                    ),
                    (
                        "type",
                        output.type.value,
                    ),
                    (
                        "domain",
                        str(
                            record.metadata.get(
                                "domain",
                                "unknown",
                            )
                        ),
                    ),
                }
                for group in groups:
                    grouped_probs[
                        group
                    ].append(
                        probability
                    )
                    grouped_targets[
                        group
                    ].append(
                        target_value
                    )

    normal_metrics = _metrics_dict(
        normal_probs,
        targets,
    )
    shuffled_metrics = _metrics_dict(
        shuffled_probs,
        targets,
    )
    uniform_metrics = _metrics_dict(
        uniform_probs,
        targets,
    )

    grouped: dict[
        str,
        dict[str, object],
    ] = defaultdict(dict)
    for (
        group_type,
        name,
    ), probabilities in sorted(
        grouped_probs.items()
    ):
        grouped[
            group_type
        ][name] = _metrics_dict(
            probabilities,
            grouped_targets[
                (
                    group_type,
                    name,
                )
            ],
        )

    total_seconds = (
        sum(latencies_ms)
        / 1000.0
    )
    return {
        "checkpoint": str(
            Path(checkpoint)
        ),
        "data": dataset_manifest(
            data
        ),
        "temperature": temperature,
        "device": str(device),
        "model": {
            "head_kind": model.head_kind,
            "head_rank": model.head_rank,
            "backbone": model.backbone_name,
        },
        "normal": normal_metrics,
        "shuffled_state": (
            shuffled_metrics
        ),
        "uniform": uniform_metrics,
        "controls": {
            "accuracy_delta_vs_shuffled": (
                float(
                    normal_metrics[
                        "accuracy"
                    ]
                )
                - float(
                    shuffled_metrics[
                        "accuracy"
                    ]
                )
            ),
            "accuracy_delta_vs_uniform": (
                float(
                    normal_metrics[
                        "accuracy"
                    ]
                )
                - float(
                    uniform_metrics[
                        "accuracy"
                    ]
                )
            ),
            "mean_normal_vs_shuffled_kl": (
                float(
                    np.mean(
                        sensitivity
                    )
                )
                if sensitivity
                else 0.0
            ),
            "choice_order_invariance": {
                "decisions": len(
                    choice_order_top1
                ),
                "top1_agreement": (
                    float(
                        np.mean(
                            choice_order_top1
                        )
                    )
                    if choice_order_top1
                    else 1.0
                ),
                "mean_abs_probability_delta": (
                    float(
                        np.mean(
                            choice_order_abs_delta
                        )
                    )
                    if choice_order_abs_delta
                    else 0.0
                ),
                "max_abs_probability_delta": (
                    float(
                        np.max(
                            choice_order_abs_delta
                        )
                    )
                    if choice_order_abs_delta
                    else 0.0
                ),
                "mean_kl": (
                    float(
                        np.mean(
                            choice_order_kl
                        )
                    )
                    if choice_order_kl
                    else 0.0
                ),
            },
        },
        "latency": {
            "batch_size": batch_size,
            "batches": len(
                latencies_ms
            ),
            "states": state_count,
            "decisions": decision_count,
            "batch_ms_median": (
                median(
                    latencies_ms
                )
                if latencies_ms
                else 0.0
            ),
            "batch_ms_p95": (
                _percentile(
                    latencies_ms,
                    0.95,
                )
            ),
            "states_per_second": (
                state_count
                / total_seconds
                if total_seconds > 0
                else 0.0
            ),
            "decisions_per_second": (
                decision_count
                / total_seconds
                if total_seconds > 0
                else 0.0
            ),
        },
        "by": dict(grouped),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark calibration, state sensitivity, "
            "Choice-order invariance, and throughput "
            "for a my-jev checkpoint"
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--data",
        required=True,
    )
    parser.add_argument(
        "--calibration"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--output",
        help=(
            "Optional JSON output path"
        ),
    )
    args = parser.parse_args()

    result = benchmark(
        args.checkpoint,
        args.data,
        calibration=args.calibration,
        batch_size=args.batch_size,
    )
    text = json.dumps(
        result,
        indent=2,
        sort_keys=True,
    )
    print(text)
    if args.output:
        output = Path(
            args.output
        )
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output.write_text(
            text + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
