from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .baseline_evidence import validate_baseline_run
from .review import ReviewWeights, build_review_queue, load_review_records
from .data import dump_jsonl
from .manifest import write_manifest
from .server import PolicyRuntime
from .shadow_replay import replay_export

BUNDLE_VERSION = "assistx-frozen-shadow-evaluation-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def run_shadow_evaluation(
    *,
    run_dir: str | Path,
    shadow_export: str | Path,
    output_dir: str | Path | None = None,
    expected_sha: str | None = None,
    device: str | None = None,
    limit: int | None = 200,
    min_priority: float = 0.25,
    weights: ReviewWeights = ReviewWeights(),
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    source = Path(shadow_export).resolve()
    if not source.is_file():
        raise ValueError(f"shadow export does not exist: {source}")

    baseline = validate_baseline_run(root, expected_sha=expected_sha)
    candidate_sha = str(baseline["git_sha"])
    checkpoint = root / "checkpoints/best"
    calibration = root / "calibration.json"

    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "shadow-evaluation"
    )
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError(
            f"shadow evaluation directory must be empty: {destination}"
        )

    baseline_validation = destination / "baseline-validation.json"
    baseline_validation.write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    runtime = PolicyRuntime.load(
        str(checkpoint),
        calibration=str(calibration),
        device=device,
    )
    replay = destination / "shadow-replay.jsonl"
    replay_manifest = replay_export(
        input_path=source,
        output_path=replay,
        runtime=runtime,
    )

    records = load_review_records(replay)
    queue = build_review_queue(
        records,
        limit=limit,
        min_priority=min_priority,
        weights=weights,
    )
    review = destination / "shadow-review-queue.jsonl"
    if queue:
        dump_jsonl(iter(queue), review)
    else:
        review.write_text("", encoding="utf-8")
    review_manifest_path = review.with_suffix(review.suffix + ".manifest.json")
    review_manifest = write_manifest(review, review_manifest_path)

    review_summary_path = review.with_suffix(review.suffix + ".review.json")
    review_summary = {
        "version": BUNDLE_VERSION,
        "evidence_only": True,
        "dispatch_allowed": False,
        "input": _artifact(replay),
        "weights": {
            "correction": weights.correction,
            "inconsistency": weights.inconsistency,
            "disagreement": weights.disagreement,
            "uncertainty": weights.uncertainty,
        },
        "min_priority": min_priority,
        "limit": limit,
        "records_considered": len(records),
        "records_queued": len(queue),
    }
    review_summary_path.write_text(
        json.dumps(review_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    receipt = {
        "version": BUNDLE_VERSION,
        "evidence_only": True,
        "dispatch_allowed": False,
        "runtime_authority_changed": False,
        "candidate": {
            "git_sha": candidate_sha,
            "checkpoint": _artifact(checkpoint / "model.pt"),
            "checkpoint_config": _artifact(checkpoint / "my_jev_config.json"),
            "calibration": _artifact(calibration),
            "experiment_manifest": _artifact(root / "manifest.json"),
            "baseline_receipt": _artifact(root / "r9700-baseline-receipt.json"),
        },
        "shadow_export": _artifact(source),
        "replay": {
            "artifact": _artifact(replay),
            "manifest": _artifact(replay.with_suffix(replay.suffix + ".manifest.json")),
            "summary": replay_manifest.get("summary", {}),
        },
        "review": {
            "artifact": _artifact(review),
            "manifest": _artifact(review_manifest_path),
            "summary": _artifact(review_summary_path),
            "records_queued": len(queue),
            "manifest_records": review_manifest.get("records"),
        },
        "authority": {
            "observer_only": True,
            "may_dispatch": False,
            "may_mutate_tasks": False,
            "may_grant_permissions": False,
            "may_satisfy_approvals": False,
            "may_override_resolver": False,
        },
    }
    receipt_path = destination / "shadow-evaluation-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an exact-SHA R9700 candidate and produce a frozen, "
            "non-dispatching AssistX shadow replay/review evidence bundle"
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--shadow-export", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-sha")
    parser.add_argument("--device")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--min-priority", type=float, default=0.25)
    args = parser.parse_args()

    receipt = run_shadow_evaluation(
        run_dir=args.run_dir,
        shadow_export=args.shadow_export,
        output_dir=args.output_dir,
        expected_sha=args.expected_sha,
        device=args.device,
        limit=None if args.limit == 0 else args.limit,
        min_priority=args.min_priority,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
