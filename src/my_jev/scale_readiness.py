from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baseline_evidence import validate_baseline_run


@dataclass(frozen=True)
class ScaleReadinessThresholds:
    min_accuracy: float = 0.65
    max_ece: float = 0.15
    max_nll: float = 1.25
    max_brier: float = 0.45
    min_accuracy_delta_vs_shuffled: float = 0.08
    min_accuracy_delta_vs_uniform: float = 0.10
    min_normal_vs_shuffled_kl: float = 0.02
    min_choice_order_top1_agreement: float = 0.98
    max_choice_order_mean_abs_delta: float = 0.02
    max_policy_consistency_violation_rate: float = 0.03


def _load_json(
    path: Path,
) -> dict[str, Any]:
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
            f"{path} must contain a JSON object"
        )
    return payload


def _metric(
    payload: dict[str, Any],
    *path: str,
) -> float:
    current: Any = payload
    for key in path:
        if not isinstance(
            current,
            dict,
        ):
            raise ValueError(
                "missing benchmark metric: "
                + ".".join(path)
            )
        if key not in current:
            raise ValueError(
                "missing benchmark metric: "
                + ".".join(path)
            )
        current = current[key]

    value = float(
        current
    )
    if not math.isfinite(
        value
    ):
        raise ValueError(
            "non-finite benchmark metric: "
            + ".".join(path)
        )
    return value


def _minimum_gate(
    name: str,
    actual: float,
    threshold: float,
) -> dict[str, object]:
    return {
        "name": name,
        "actual": actual,
        "operator": ">=",
        "threshold": threshold,
        "passed": (
            actual >= threshold
        ),
    }


def _maximum_gate(
    name: str,
    actual: float,
    threshold: float,
) -> dict[str, object]:
    return {
        "name": name,
        "actual": actual,
        "operator": "<=",
        "threshold": threshold,
        "passed": (
            actual <= threshold
        ),
    }


def evaluate_scale_readiness(
    run_dir: str | Path,
    *,
    expected_sha: str | None = None,
    thresholds: ScaleReadinessThresholds = (
        ScaleReadinessThresholds()
    ),
) -> dict[str, object]:
    """Decide whether a small exact-SHA baseline justifies the full run.

    This is a resource-allocation gate only. It never grants Hermes/AssistX
    routing, mutation, approval, tool, or dispatch authority.
    """

    root = Path(
        run_dir
    ).resolve()

    evidence = (
        validate_baseline_run(
            root,
            expected_sha=(
                expected_sha
            ),
        )
    )
    manifest = _load_json(
        root
        / "manifest.json"
    )
    benchmark = _load_json(
        root
        / "benchmark.json"
    )
    promotion = _load_json(
        root
        / "promotion.json"
    )
    fleet = _load_json(
        root
        / "fleet-benchmark.json"
    )

    spec = manifest.get(
        "spec"
    )
    if not isinstance(
        spec,
        dict,
    ):
        raise ValueError(
            "baseline manifest has no experiment spec"
        )
    experiment = spec.get(
        "experiment"
    )
    if not isinstance(
        experiment,
        dict,
    ):
        raise ValueError(
            "baseline manifest has no experiment identity"
        )
    experiment_name = str(
        experiment.get(
            "name"
        )
        or ""
    )
    expected_experiment = (
        "assistx-policy-modernbert-r9700-baseline"
    )
    if (
        experiment_name
        != expected_experiment
    ):
        raise ValueError(
            "scale readiness requires the pinned "
            "R9700 ModernBERT baseline experiment"
        )

    audit = manifest.get(
        "data_audit"
    )
    audit_passed = bool(
        isinstance(
            audit,
            dict,
        )
        and audit.get(
            "passed"
        )
    )

    gates = [
        _minimum_gate(
            "min_accuracy",
            _metric(
                benchmark,
                "normal",
                "accuracy",
            ),
            thresholds.min_accuracy,
        ),
        _maximum_gate(
            "max_ece",
            _metric(
                benchmark,
                "normal",
                "ece",
            ),
            thresholds.max_ece,
        ),
        _maximum_gate(
            "max_nll",
            _metric(
                benchmark,
                "normal",
                "nll",
            ),
            thresholds.max_nll,
        ),
        _maximum_gate(
            "max_brier",
            _metric(
                benchmark,
                "normal",
                "brier",
            ),
            thresholds.max_brier,
        ),
        _minimum_gate(
            "min_accuracy_delta_vs_shuffled",
            _metric(
                benchmark,
                "controls",
                "accuracy_delta_vs_shuffled",
            ),
            thresholds.min_accuracy_delta_vs_shuffled,
        ),
        _minimum_gate(
            "min_accuracy_delta_vs_uniform",
            _metric(
                benchmark,
                "controls",
                "accuracy_delta_vs_uniform",
            ),
            thresholds.min_accuracy_delta_vs_uniform,
        ),
        _minimum_gate(
            "min_normal_vs_shuffled_kl",
            _metric(
                benchmark,
                "controls",
                "mean_normal_vs_shuffled_kl",
            ),
            thresholds.min_normal_vs_shuffled_kl,
        ),
        _minimum_gate(
            "min_choice_order_top1_agreement",
            _metric(
                benchmark,
                "controls",
                "choice_order_invariance",
                "top1_agreement",
            ),
            thresholds.min_choice_order_top1_agreement,
        ),
        _maximum_gate(
            "max_choice_order_mean_abs_delta",
            _metric(
                benchmark,
                "controls",
                "choice_order_invariance",
                "mean_abs_probability_delta",
            ),
            thresholds.max_choice_order_mean_abs_delta,
        ),
        _maximum_gate(
            "max_policy_consistency_violation_rate",
            _metric(
                benchmark,
                "controls",
                "policy_consistency",
                "violation_rate",
            ),
            thresholds.max_policy_consistency_violation_rate,
        ),
    ]

    structural = [
        {
            "name": (
                "validated_exact_sha_evidence"
            ),
            "passed": bool(
                evidence.get(
                    "git_sha"
                )
            ),
        },
        {
            "name": (
                "pretraining_data_audit"
            ),
            "passed": (
                audit_passed
            ),
        },
    ]

    failed = [
        str(
            gate["name"]
        )
        for gate in gates
        if not gate["passed"]
    ]
    failed.extend(
        str(
            gate["name"]
        )
        for gate in structural
        if not gate["passed"]
    )

    fleet_promotion = (
        fleet.get(
            "promotion"
        )
        or {}
    )
    if not isinstance(
        fleet_promotion,
        dict,
    ):
        fleet_promotion = {}

    shadow_receipt_path = (
        root
        / "shadow-evaluation"
        / "shadow-evaluation-receipt.json"
    )
    shadow: dict[str, object]
    if shadow_receipt_path.is_file():
        shadow_receipt = _load_json(
            shadow_receipt_path
        )
        candidate = (
            shadow_receipt.get(
                "candidate"
            )
            or {}
        )
        candidate_sha = (
            candidate.get(
                "git_sha"
            )
            if isinstance(
                candidate,
                dict,
            )
            else None
        )
        shadow = {
            "completed": True,
            "same_candidate_sha": (
                str(
                    candidate_sha
                    or ""
                )
                == str(
                    evidence[
                        "git_sha"
                    ]
                )
            ),
            "dispatch_allowed": bool(
                shadow_receipt.get(
                    "dispatch_allowed",
                    False,
                )
            ),
            "runtime_authority_changed": bool(
                shadow_receipt.get(
                    "runtime_authority_changed",
                    False,
                )
            ),
        }
    else:
        shadow = {
            "completed": False,
            "reason": (
                "shadow evaluation is optional "
                "for synthetic scale-up readiness"
            ),
        }

    passed = not failed

    return {
        "schema_version": 1,
        "kind": (
            "r9700-modernbert-scale-readiness"
        ),
        "decision": (
            "scale"
            if passed
            else "hold"
        ),
        "passed": passed,
        "failed": failed,
        "git_sha": evidence[
            "git_sha"
        ],
        "baseline_run_dir": str(
            root
        ),
        "experiment": (
            experiment_name
        ),
        "gates": gates,
        "structural_checks": structural,
        "baseline_promotion": {
            "passed": bool(
                promotion.get(
                    "passed",
                    False,
                )
            ),
            "failed": (
                promotion.get(
                    "failed",
                    []
                )
            ),
            "note": (
                "baseline promotion may include "
                "exploratory fleet checks and is "
                "not itself the scale-up decision"
            ),
        },
        "fleet_observer_evidence": {
            "blocking": False,
            "passed": bool(
                fleet_promotion.get(
                    "passed",
                    False,
                )
            ),
            "failed": (
                fleet_promotion.get(
                    "failed",
                    []
                )
            ),
        },
        "shadow_evaluation": shadow,
        "next_experiment": (
            "configs/experiments/"
            "assistx-modernbert-r9700-scaleup.toml"
        ),
        "authority_boundary": {
            "evidence_only": True,
            "runtime_authority_changed": False,
            "dispatch_allowed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether an exact-SHA "
            "R9700 small baseline justifies "
            "the larger ModernBERT run"
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
    )
    parser.add_argument(
        "--expected-sha",
    )
    parser.add_argument(
        "--output",
    )
    args = parser.parse_args()

    result = (
        evaluate_scale_readiness(
            args.run_dir,
            expected_sha=(
                args.expected_sha
            ),
        )
    )
    rendered = (
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    if args.output:
        output = Path(
            args.output
        )
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output.write_text(
            rendered,
            encoding="utf-8",
        )

    print(
        rendered,
        end="",
    )

    if not result[
        "passed"
    ]:
        raise SystemExit(
            2
        )


if __name__ == "__main__":
    main()
