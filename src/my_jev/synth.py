from __future__ import annotations

import argparse
import random
from math import exp, floor
from pathlib import Path

from .data import dump_jsonl
from .manifest import write_manifest
from .schema import DecisionRecord, QuestionSpec, QuestionType, TargetSpec

GENERATOR_VERSION = "system-one-synth-v1"
DOMAINS = ("operations", "support", "build")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def _binary_distribution(probability_true: float) -> TargetSpec:
    probability_true = _clamp(probability_true, 0.001, 0.999)
    return TargetSpec(
        distribution=[1.0 - probability_true, probability_true]
    )


def _score_distribution(value: float, levels: int) -> TargetSpec:
    value = _clamp(value, 0.0, float(levels - 1))
    low = int(floor(value))
    high = min(levels - 1, low + 1)
    distribution = [0.0] * levels
    if low == high:
        distribution[low] = 1.0
    else:
        high_mass = value - low
        distribution[low] = 1.0 - high_mass
        distribution[high] = high_mass
    return TargetSpec(distribution=distribution)


def _question(
    rng: random.Random,
    *,
    question_type: QuestionType,
    task_id: str,
    instructions: tuple[str, ...],
    options: list[str] | None = None,
) -> QuestionSpec:
    variant = rng.randrange(len(instructions))
    return QuestionSpec(
        type=question_type,
        instructions=instructions[variant],
        options=options,
        metadata={
            "task_id": task_id,
            "schema_variant": variant,
        },
    )


def _choice_question(
    rng: random.Random,
    *,
    task_id: str,
    instructions: tuple[str, ...],
    options: list[str],
    correct: str,
) -> tuple[QuestionSpec, TargetSpec]:
    shuffled = list(options)
    rng.shuffle(shuffled)
    question = _question(
        rng,
        question_type=QuestionType.CHOICE,
        task_id=task_id,
        instructions=instructions,
        options=shuffled,
    )
    return question, TargetSpec(index=shuffled.index(correct))


def _record(
    *,
    state: str,
    questions: dict[str, QuestionSpec],
    targets: dict[str, TargetSpec],
    metadata: dict[str, object],
) -> DecisionRecord:
    return DecisionRecord(
        state=state,
        questions=questions,
        targets=targets,
        metadata={
            **metadata,
            "generator_version": GENERATOR_VERSION,
            "label_source": "synthetic_verifier",
        },
    )


def _operations_record(
    rng: random.Random,
    *,
    family_id: str,
    record_id: str,
    counterfactual: bool,
) -> DecisionRecord:
    service = rng.choice(
        ["checkout", "identity", "search", "notifications", "ingest"]
    )
    fault = rng.choice(["application", "database", "network"])
    impact = rng.uniform(0.05, 0.85)
    duration = rng.randint(3, 105)
    data_loss = rng.random() < 0.08

    if counterfactual:
        impact = 0.92 if impact < 0.55 else 0.14

    clue = {
        "application": rng.choice(
            [
                "The issue began immediately after a new application deploy.",
                "Stack traces point to an exception in the service code.",
                "Only the newest application instances show the failure.",
            ]
        ),
        "database": rng.choice(
            [
                "Database lock waits and connection-pool saturation are elevated.",
                "Queries are timing out while database CPU is saturated.",
                "The database reports a sharp increase in blocked transactions.",
            ]
        ),
        "network": rng.choice(
            [
                "Packet loss between availability zones is elevated.",
                "Cross-zone connections are resetting before reaching the service.",
                "Network telemetry shows intermittent route loss.",
            ]
        ),
    }[fault]

    statements = [
        f"The {service} service has been degraded for {duration} minutes.",
        f"About {round(impact * 100)}% of requests are affected.",
        clue,
        (
            "Confirmed writes may have been lost."
            if data_loss
            else "There is no evidence of lost writes."
        ),
    ]
    rng.shuffle(statements)
    state = " ".join(statements)

    severity_value = _clamp(
        impact * 2.5
        + min(duration / 120.0, 0.45)
        + (0.8 if data_loss else 0.0),
        0.0,
        3.0,
    )
    urgent = bool(
        severity_value >= 1.75
        or data_loss
        or impact >= 0.75
    )
    escalation_probability = _clamp(
        _sigmoid(
            -2.5
            + 4.0 * impact
            + 1.1 * float(data_loss)
            + duration / 80.0
        ),
        0.02,
        0.98,
    )

    owner_question, owner_target = _choice_question(
        rng,
        task_id="ops.owner",
        instructions=(
            "Which engineering area should own the next action?",
            "Who is the best primary owner for this incident?",
            "Route this incident to the most likely responsible area.",
        ),
        options=["application", "database", "network", "unknown"],
        correct=fault,
    )

    questions = {
        "urgent": _question(
            rng,
            question_type=QuestionType.NOUL,
            task_id="ops.urgent",
            instructions=(
                "Does this incident require immediate human attention?",
                "Should an engineer respond to this incident right now?",
                "Is immediate operational intervention warranted?",
            ),
        ),
        "owner": owner_question,
        "severity": _question(
            rng,
            question_type=QuestionType.SCORE,
            task_id="ops.severity",
            instructions=(
                "How severe is the current customer impact?",
                "Rate the operational severity of this incident.",
                "What is the present incident severity?",
            ),
            options=["minor", "moderate", "major", "critical"],
        ),
        "escalates": _question(
            rng,
            question_type=QuestionType.NOUL,
            task_id="ops.escalates",
            instructions=(
                "Will this incident escalate within the next 30 minutes?",
                "How likely is this incident to worsen within 30 minutes?",
                "Is the incident expected to escalate in the next half hour?",
            ),
        ),
    }
    targets = {
        "urgent": TargetSpec(index=int(urgent)),
        "owner": owner_target,
        "severity": _score_distribution(severity_value, 4),
        "escalates": _binary_distribution(escalation_probability),
    }
    return _record(
        state=state,
        questions=questions,
        targets=targets,
        metadata={
            "domain": "operations",
            "family_id": family_id,
            "record_id": record_id,
            "counterfactual": counterfactual,
            "latent_fault": fault,
        },
    )


def _support_record(
    rng: random.Random,
    *,
    family_id: str,
    record_id: str,
    counterfactual: bool,
) -> DecisionRecord:
    topic = rng.choice(["billing", "account", "technical", "fulfillment"])
    waiting_days = rng.randint(0, 6)
    frustration = rng.uniform(0.05, 0.95)
    premium = rng.random() < 0.25
    outage = topic == "technical" and rng.random() < 0.35

    if counterfactual:
        waiting_days = 8 if waiting_days < 3 else 0

    clue = {
        "billing": "The customer disputes a charge and asks for a billing correction.",
        "account": "The customer cannot regain access after an identity check.",
        "technical": (
            "The product fails during normal use and the customer supplied diagnostics."
        ),
        "fulfillment": "The customer says the shipment has not arrived as expected.",
    }[topic]

    statements = [
        clue,
        f"The case has been waiting for {waiting_days} days.",
        (
            "The customer is on the premium plan."
            if premium
            else "The customer is on the standard plan."
        ),
        (
            "A known service outage affects the reported workflow."
            if outage
            else "No matching service outage is currently known."
        ),
        (
            "The message is highly frustrated."
            if frustration >= 0.65
            else "The message is calm and factual."
        ),
    ]
    rng.shuffle(statements)
    state = " ".join(statements)

    route_map = {
        "billing": "billing_ops",
        "account": "identity",
        "technical": "technical_support",
        "fulfillment": "fulfillment",
    }
    priority_value = _clamp(
        0.35 * waiting_days
        + 1.15 * frustration
        + (0.8 if premium else 0.0)
        + (0.9 if outage else 0.0),
        0.0,
        3.0,
    )
    urgent = bool(
        priority_value >= 2.0
        or outage
        or waiting_days >= 6
    )
    churn_probability = _clamp(
        _sigmoid(
            -2.4
            + 2.8 * frustration
            + 0.23 * waiting_days
            + 0.55 * float(outage)
            + 0.35 * float(premium)
        ),
        0.02,
        0.98,
    )

    owner_question, owner_target = _choice_question(
        rng,
        task_id="support.owner",
        instructions=(
            "Which team should own the next customer action?",
            "Route this case to the correct support function.",
            "Who should take primary ownership of this case?",
        ),
        options=[
            "billing_ops",
            "identity",
            "technical_support",
            "fulfillment",
        ],
        correct=route_map[topic],
    )

    questions = {
        "urgent": _question(
            rng,
            question_type=QuestionType.NOUL,
            task_id="support.urgent",
            instructions=(
                "Does this case need immediate handling?",
                "Should this support case be acted on now?",
                "Is immediate intervention warranted for this customer?",
            ),
        ),
        "owner": owner_question,
        "severity": _question(
            rng,
            question_type=QuestionType.SCORE,
            task_id="support.priority",
            instructions=(
                "What priority should this case receive?",
                "Rate the urgency and customer impact of this case.",
                "How high should this case rank in the support queue?",
            ),
            options=["low", "normal", "high", "urgent"],
        ),
        "escalates": _question(
            rng,
            question_type=QuestionType.NOUL,
            task_id="support.churn",
            instructions=(
                "Will this customer churn within 30 days if the issue persists?",
                "Is the customer likely to leave within 30 days without resolution?",
                "Does this case carry near-term churn risk?",
            ),
        ),
    }
    targets = {
        "urgent": TargetSpec(index=int(urgent)),
        "owner": owner_target,
        "severity": _score_distribution(priority_value, 4),
        "escalates": _binary_distribution(churn_probability),
    }
    return _record(
        state=state,
        questions=questions,
        targets=targets,
        metadata={
            "domain": "support",
            "family_id": family_id,
            "record_id": record_id,
            "counterfactual": counterfactual,
            "latent_topic": topic,
        },
    )


def _build_record(
    rng: random.Random,
    *,
    family_id: str,
    record_id: str,
    counterfactual: bool,
) -> DecisionRecord:
    cause = rng.choice(
        ["dependency", "test", "compiler", "infrastructure"]
    )
    release_branch = rng.random() < 0.35
    repeats = rng.randint(1, 5)
    flaky = cause == "test" and rng.random() < 0.55

    if counterfactual:
        release_branch = not release_branch

    clue = {
        "dependency": "Package resolution failed because a required artifact was unavailable.",
        "test": (
            "The build completed but a test suite failed intermittently."
            if flaky
            else "The build completed but a deterministic test assertion failed."
        ),
        "compiler": "Compilation stopped on a type or syntax error in changed code.",
        "infrastructure": "The runner lost its executor or remote cache connection.",
    }[cause]

    statements = [
        clue,
        f"The same failure has occurred {repeats} time(s).",
        (
            "This build gates a release branch."
            if release_branch
            else "This build is from a non-release development branch."
        ),
        (
            "The failing test has a recent flaky history."
            if flaky
            else "There is no matching flaky-test history."
        ),
    ]
    rng.shuffle(statements)
    state = " ".join(statements)

    block_release = bool(
        release_branch
        and (
            cause != "test"
            or not flaky
            or repeats >= 3
        )
    )
    severity_value = _clamp(
        0.45 * repeats
        + (1.25 if release_branch else 0.0)
        + (0.55 if cause == "infrastructure" else 0.0)
        - (0.45 if flaky else 0.0),
        0.0,
        3.0,
    )
    rerun_probability = {
        "dependency": 0.12,
        "test": 0.78 if flaky else 0.08,
        "compiler": 0.03,
        "infrastructure": 0.58,
    }[cause]
    rerun_probability = _clamp(
        rerun_probability - 0.08 * (repeats - 1),
        0.01,
        0.97,
    )

    owner_question, owner_target = _choice_question(
        rng,
        task_id="build.cause",
        instructions=(
            "What is the most likely primary cause of this build failure?",
            "Classify the failure into its responsible category.",
            "Which failure class best explains this build?",
        ),
        options=["dependency", "test", "compiler", "infrastructure"],
        correct=cause,
    )

    questions = {
        "urgent": _question(
            rng,
            question_type=QuestionType.NOUL,
            task_id="build.block_release",
            instructions=(
                "Should this failure block the release?",
                "Does this build failure require stopping the release?",
                "Is this failure release-blocking?",
            ),
        ),
        "owner": owner_question,
        "severity": _question(
            rng,
            question_type=QuestionType.SCORE,
            task_id="build.severity",
            instructions=(
                "How severe is this build failure?",
                "Rate the operational impact of this build failure.",
                "What severity should be assigned to this failure?",
            ),
            options=["minor", "moderate", "major", "critical"],
        ),
        "escalates": _question(
            rng,
            question_type=QuestionType.NOUL,
            task_id="build.rerun_passes",
            instructions=(
                "Will an unchanged rerun pass?",
                "Is a rerun likely to succeed without a code change?",
                "Would retrying this build unchanged probably succeed?",
            ),
        ),
    }
    targets = {
        "urgent": TargetSpec(index=int(block_release)),
        "owner": owner_target,
        "severity": _score_distribution(severity_value, 4),
        "escalates": _binary_distribution(rerun_probability),
    }
    return _record(
        state=state,
        questions=questions,
        targets=targets,
        metadata={
            "domain": "build",
            "family_id": family_id,
            "record_id": record_id,
            "counterfactual": counterfactual,
            "latent_cause": cause,
        },
    )


_GENERATORS = {
    "operations": _operations_record,
    "support": _support_record,
    "build": _build_record,
}


def generate_synthetic_records(
    count: int,
    *,
    seed: int = 17,
) -> list[DecisionRecord]:
    if count < 1:
        raise ValueError("count must be >= 1")

    root = random.Random(seed)
    records: list[DecisionRecord] = []
    family_count = (count + 1) // 2

    for family_index in range(family_count):
        domain = DOMAINS[family_index % len(DOMAINS)]
        family_seed = root.randrange(0, 2**63)
        family_id = f"{domain}-{family_index:08d}"

        for counterfactual in (False, True):
            if len(records) >= count:
                break
            local = random.Random(family_seed)
            record_id = (
                f"{family_id}-"
                f"{'counterfactual' if counterfactual else 'base'}"
            )
            records.append(
                _GENERATORS[domain](
                    local,
                    family_id=family_id,
                    record_id=record_id,
                    counterfactual=counterfactual,
                )
            )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic verifier-labeled decision corpus"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = generate_synthetic_records(
        args.records,
        seed=args.seed,
    )
    dump_jsonl(iter(records), output)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest = write_manifest(output, manifest_path)
    print(
        f"wrote {manifest['records']} records, "
        f"{manifest['questions']} decisions, "
        f"sha256={manifest['sha256']}"
    )
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
