from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .calibration import load_temperature
from .checkpoint import load_checkpoint
from .data import load_jsonl
from .fleet_eval import (
    FleetGateThresholds,
    evaluate_fleet_promotion,
    evaluate_fleet_safety,
)
from .fleet_family_eval import (
    evaluate_fleet_families,
)
from .fleet_policy import (
    FleetPlacementState,
    PlacementShape,
    build_fleet_placement_record,
)
from .schema import DecisionRecord


def _sha256(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_states(
    path: str | Path,
) -> list[FleetPlacementState]:
    source = Path(path)
    states = []
    for line in source.read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            states.append(
                FleetPlacementState
                .model_validate_json(
                    line
                )
            )
    if not states:
        raise ValueError(
            "fleet benchmark requires "
            "at least one state"
        )
    return states


def load_family_records(
    path: str | Path,
) -> list[DecisionRecord]:
    records = load_jsonl(
        path
    )
    if not records:
        raise ValueError(
            "fleet family benchmark "
            "requires at least one record"
        )

    for index, record in enumerate(
        records
    ):
        if (
            record.metadata.get(
                "domain"
            )
            != "fleet_placement"
        ):
            raise ValueError(
                "fleet family benchmark "
                f"record {index} has "
                "wrong domain"
            )
        if (
            "family_id"
            not in record.metadata
        ):
            raise ValueError(
                "fleet family benchmark "
                f"record {index} has no "
                "family_id"
            )
        if not record.targets:
            raise ValueError(
                "fleet family benchmark "
                f"record {index} is "
                "unlabeled"
            )
    return records


def _predict_family_records(
    model,
    records: list[DecisionRecord],
    *,
    temperature: float,
    batch_size: int,
) -> list[
    dict[str, dict[str, float]]
]:
    if batch_size < 1:
        raise ValueError(
            "family batch size must be >= 1"
        )

    predictions: list[
        dict[str, dict[str, float]]
    ] = [
        {}
        for _ in records
    ]

    for start in range(
        0,
        len(records),
        batch_size,
    ):
        batch = records[
            start : start + batch_size
        ]
        outputs = model.forward_records(
            batch
        )
        for output in outputs:
            probabilities = (
                output
                .probabilities_at_temperature(
                    temperature
                )
                .detach()
                .cpu()
                .tolist()
            )
            predictions[
                start
                + output.record_index
            ][output.name] = {
                str(option): float(
                    probability
                )
                for option, probability in zip(
                    output.options,
                    probabilities,
                    strict=True,
                )
            }

    return predictions


def fleet_benchmark(
    checkpoint: str,
    states_path: str,
    *,
    calibration: str | None = None,
    family_data: str | None = None,
    family_batch_size: int = 8,
    thresholds: FleetGateThresholds = (
        FleetGateThresholds()
    ),
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

    def predict(
        state: FleetPlacementState,
    ) -> PlacementShape:
        record = (
            build_fleet_placement_record(
                state
            )
        )
        outputs = model.forward_records(
            [record]
        )
        placement = next(
            output
            for output in outputs
            if output.name
            == "placement"
        )
        probabilities = (
            placement
            .probabilities_at_temperature(
                temperature
            )
        )
        index = int(
            probabilities
            .argmax()
            .item()
        )
        return PlacementShape(
            placement.options[
                index
            ]
        )

    states = load_states(
        states_path
    )
    family_records: (
        list[DecisionRecord] | None
    ) = None
    family_metrics: (
        dict[str, object] | None
    ) = None

    with torch.inference_mode():
        metrics = evaluate_fleet_safety(
            states,
            predict,
        )

        if family_data:
            family_records = (
                load_family_records(
                    family_data
                )
            )
            family_predictions = (
                _predict_family_records(
                    model,
                    family_records,
                    temperature=(
                        temperature
                    ),
                    batch_size=(
                        family_batch_size
                    ),
                )
            )
            family_metrics = (
                evaluate_fleet_families(
                    family_records,
                    family_predictions,
                )
            )

    promotion = evaluate_fleet_promotion(
        metrics,
        thresholds,
        family_metrics=family_metrics,
    )

    source = Path(
        states_path
    )
    family_source = (
        Path(family_data)
        if family_data
        else None
    )

    return {
        "schema_version": 2,
        "checkpoint": str(
            Path(checkpoint)
        ),
        "calibration": calibration,
        "temperature": temperature,
        "device": str(device),
        "states": {
            "path": str(source),
            "sha256": _sha256(
                source
            ),
            "count": len(
                states
            ),
        },
        "family_data": (
            {
                "path": str(
                    family_source
                ),
                "sha256": _sha256(
                    family_source
                ),
                "count": len(
                    family_records
                    or []
                ),
            }
            if family_source
            is not None
            else None
        ),
        "metrics": metrics,
        "family_metrics": (
            family_metrics
        ),
        "promotion": promotion,
        "authority_boundary": (
            "advisory_only"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a checkpoint "
            "against fleet safety and "
            "family counterfactuals"
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--states",
        required=True,
    )
    parser.add_argument(
        "--calibration",
    )
    parser.add_argument(
        "--family-data",
    )
    parser.add_argument(
        "--family-batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--min-hard-failure-defer-rate",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--min-capacity-boundary-accuracy",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--min-node-permutation-agreement",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--min-pressure-sensitivity-rate",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--min-family-invariant-top1-agreement",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        (
            "--max-family-invariant-"
            "mean-abs-probability-delta"
        ),
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--min-family-semantic-new-target-accuracy",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        (
            "--min-family-semantic-"
            "probability-direction-rate"
        ),
        type=float,
        default=0.80,
    )
    parser.add_argument(
        (
            "--min-family-semantic-"
            "unchanged-top1-agreement"
        ),
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--min-family-success-rate",
        type=float,
        default=0.75,
    )
    args = parser.parse_args()

    thresholds = FleetGateThresholds(
        min_hard_failure_defer_rate=(
            args
            .min_hard_failure_defer_rate
        ),
        min_capacity_boundary_accuracy=(
            args
            .min_capacity_boundary_accuracy
        ),
        min_node_permutation_agreement=(
            args
            .min_node_permutation_agreement
        ),
        min_pressure_sensitivity_rate=(
            args
            .min_pressure_sensitivity_rate
        ),
        min_family_invariant_top1_agreement=(
            args
            .min_family_invariant_top1_agreement
        ),
        max_family_invariant_mean_abs_probability_delta=(
            args
            .max_family_invariant_mean_abs_probability_delta
        ),
        min_family_semantic_new_target_accuracy=(
            args
            .min_family_semantic_new_target_accuracy
        ),
        min_family_semantic_probability_direction_rate=(
            args
            .min_family_semantic_probability_direction_rate
        ),
        min_family_semantic_unchanged_top1_agreement=(
            args
            .min_family_semantic_unchanged_top1_agreement
        ),
        min_family_success_rate=(
            args
            .min_family_success_rate
        ),
    )
    result = fleet_benchmark(
        args.checkpoint,
        args.states,
        calibration=(
            args.calibration
        ),
        family_data=(
            args.family_data
        ),
        family_batch_size=(
            args.family_batch_size
        ),
        thresholds=thresholds,
    )
    output = Path(
        args.output
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    rendered = json.dumps(
        result,
        indent=2,
        sort_keys=True,
    )
    output.write_text(
        rendered + "\n",
        encoding="utf-8",
    )
    print(
        rendered
    )


if __name__ == "__main__":
    main()
