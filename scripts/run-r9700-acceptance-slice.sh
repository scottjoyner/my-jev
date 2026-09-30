#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <40-char git SHA> [shadow-export-path]" >&2
  exit 64
}

EXPECTED_SHA="${1:-}"
SHADOW_EXPORT="${2:-}"
[[ "${EXPECTED_SHA}" =~ ^[0-9a-f]{40}$ ]] || usage

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

PREFLIGHT_DIR="runs/preflight/r9700/${EXPECTED_SHA}"
mkdir -p "${PREFLIGHT_DIR}"

LAUNCH_LOG="${PREFLIGHT_DIR}/acceptance-launch.log"
FAILURE_BUNDLE="${PREFLIGHT_DIR}/failure-bundle"
START_EPOCH="$(date +%s)"
CURRENT_STAGE="startup"

capture_failure() {
  local exit_code="$?"
  if [[ "${exit_code}" -eq 0 ]]; then
    return 0
  fi

  mkdir -p "${FAILURE_BUNDLE}"

  PREFLIGHT_DIR="${PREFLIGHT_DIR}" \
  FAILURE_BUNDLE="${FAILURE_BUNDLE}" \
  EXPECTED_SHA="${EXPECTED_SHA}" \
  EXIT_CODE="${exit_code}" \
  CURRENT_STAGE="${CURRENT_STAGE}" \
  START_EPOCH="${START_EPOCH}" \
  python - <<'PY' || true
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

preflight_dir = Path(
    os.environ["PREFLIGHT_DIR"]
)
bundle = Path(
    os.environ["FAILURE_BUNDLE"]
)
expected_sha = os.environ["EXPECTED_SHA"]
exit_code = int(
    os.environ["EXIT_CODE"]
)
stage = os.environ["CURRENT_STAGE"]
start_epoch = int(
    os.environ["START_EPOCH"]
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


artifacts: dict[
    str,
    dict[str, object],
] = {}

launch_log = (
    preflight_dir
    / "acceptance-launch.log"
)
if launch_log.is_file():
    target = (
        bundle
        / "acceptance-launch.log"
    )
    shutil.copy2(
        launch_log,
        target,
    )
    artifacts[
        "acceptance-launch.log"
    ] = {
        "sha256": sha256(
            target
        ),
        "bytes": (
            target.stat().st_size
        ),
    }

runs_root = Path(
    "runs/experiments"
)
matched_runs: list[
    dict[str, object]
] = []

if runs_root.is_dir():
    for run_dir in sorted(
        runs_root.iterdir()
    ):
        if not run_dir.is_dir():
            continue

        manifest_path = (
            run_dir
            / "manifest.json"
        )
        if not manifest_path.is_file():
            continue

        try:
            manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            continue

        git_state = (
            manifest.get("git")
            or {}
        )
        revision = str(
            git_state.get(
                "revision"
            )
            or ""
        )
        if revision != expected_sha:
            continue

        # Ignore stale historical attempts unless they changed near this run.
        newest_mtime = max(
            (
                path.stat().st_mtime
                for path in (
                    run_dir
                    / "manifest.json",
                    run_dir
                    / "failure.json",
                )
                if path.exists()
            ),
            default=0.0,
        )
        if (
            newest_mtime
            and newest_mtime
            < start_epoch - 60
        ):
            continue

        run_target = (
            bundle
            / run_dir.name
        )
        run_target.mkdir(
            parents=True,
            exist_ok=True,
        )

        copied: list[str] = []
        for relative in (
            "manifest.json",
            "failure.json",
            "promotion.json",
            "benchmark.json",
            "fleet-benchmark.json",
            "calibration.json",
        ):
            source = (
                run_dir
                / relative
            )
            if source.is_file():
                shutil.copy2(
                    source,
                    run_target
                    / relative,
                )
                copied.append(
                    relative
                )

        stages = (
            run_dir
            / "stages"
        )
        if stages.is_dir():
            stage_target = (
                run_target
                / "stages"
            )
            stage_target.mkdir(
                exist_ok=True
            )
            for source in stages.iterdir():
                if (
                    source.is_file()
                    and (
                        source.name.endswith(
                            ".log"
                        )
                        or source.name.endswith(
                            ".command.json"
                        )
                    )
                ):
                    shutil.copy2(
                        source,
                        stage_target
                        / source.name,
                    )
                    copied.append(
                        f"stages/{source.name}"
                    )

        matched_runs.append(
            {
                "run_dir": str(
                    run_dir
                ),
                "copied": copied,
            }
        )

receipt = {
    "schema_version": 1,
    "kind": (
        "r9700-acceptance-failure"
    ),
    "git_sha": expected_sha,
    "stage": stage,
    "exit_code": exit_code,
    "started_epoch": start_epoch,
    "captured_epoch": int(
        time.time()
    ),
    "evidence_only": True,
    "runtime_authority_changed": False,
    "dispatch_allowed": False,
    "artifacts": artifacts,
    "matched_runs": matched_runs,
}

(
    bundle
    / "acceptance-failure.json"
).write_text(
    json.dumps(
        receipt,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

  echo "R9700 acceptance failed during stage: ${CURRENT_STAGE}" >&2
  echo "failure_bundle: ${FAILURE_BUNDLE}" >&2
  return "${exit_code}"
}

trap capture_failure EXIT

CURRENT_STAGE="baseline"
bash scripts/run-r9700-modernbert-baseline.sh "${EXPECTED_SHA}"   | tee "${LAUNCH_LOG}"

RECEIPT_PATH="$(
  awk -F': ' '/^receipt: / {value=$2} END {print value}' "${LAUNCH_LOG}"
)"

if [[ -z "${RECEIPT_PATH}" ]]; then
  echo "acceptance wrapper could not locate baseline receipt path" >&2
  exit 70
fi

if [[ ! -f "${RECEIPT_PATH}" ]]; then
  echo "baseline receipt does not exist: ${RECEIPT_PATH}" >&2
  exit 71
fi

RUN_DIR="$(cd "$(dirname "${RECEIPT_PATH}")" && pwd)"

CURRENT_STAGE="evidence-validation"
EVIDENCE_JSON="${RUN_DIR}/r9700-evidence-validation.json"
python -m my_jev.baseline_evidence   --run-dir "${RUN_DIR}"   --expected-sha "${EXPECTED_SHA}"   --output "${EVIDENCE_JSON}"

CURRENT_STAGE="scale-readiness"
SCALE_READINESS_JSON="${RUN_DIR}/r9700-scale-readiness.json"
set +e
python -m my_jev.scale_readiness \
  --run-dir "${RUN_DIR}" \
  --expected-sha "${EXPECTED_SHA}" \
  --output "${SCALE_READINESS_JSON}"
SCALE_READINESS_EXIT="$?"
set -e

if [[ "${SCALE_READINESS_EXIT}" -ne 0 && "${SCALE_READINESS_EXIT}" -ne 2 ]]; then
  echo "scale-readiness evaluator failed unexpectedly: exit ${SCALE_READINESS_EXIT}" >&2
  exit "${SCALE_READINESS_EXIT}"
fi

SCALE_UP_READY="$(
  python - "${SCALE_READINESS_JSON}" <<'PY'
import json
import sys

payload = json.load(
    open(
        sys.argv[1],
        encoding="utf-8",
    )
)
print(
    "true"
    if payload.get("passed")
    else "false"
)
PY
)"

SHADOW_RECEIPT=""
if [[ -n "${SHADOW_EXPORT}" ]]; then
  if [[ ! -f "${SHADOW_EXPORT}" ]]; then
    echo "shadow export does not exist: ${SHADOW_EXPORT}" >&2
    exit 72
  fi

  SHADOW_LIMIT="${MY_JEV_SHADOW_LIMIT:-200}"
  SHADOW_MIN_PRIORITY="${MY_JEV_SHADOW_MIN_PRIORITY:-0.25}"

  CURRENT_STAGE="shadow-evaluation"
  python -m my_jev.shadow_evaluation     --run-dir "${RUN_DIR}"     --expected-sha "${EXPECTED_SHA}"     --shadow-export "${SHADOW_EXPORT}"     --device cuda     --limit "${SHADOW_LIMIT}"     --min-priority "${SHADOW_MIN_PRIORITY}"

  SHADOW_RECEIPT="${RUN_DIR}/shadow-evaluation/shadow-evaluation-receipt.json"
  if [[ ! -f "${SHADOW_RECEIPT}" ]]; then
    echo "shadow evaluation did not produce its receipt" >&2
    exit 73
  fi
fi

CURRENT_STAGE="acceptance-summary"
SUMMARY="${RUN_DIR}/r9700-acceptance-summary.json"
python -   "${RUN_DIR}"   "${EXPECTED_SHA}"   "${EVIDENCE_JSON}"   "${SCALE_READINESS_JSON}"   "${SHADOW_RECEIPT}"   > "${SUMMARY}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
expected_sha = sys.argv[2]
evidence_path = Path(sys.argv[3])
scale_readiness_path = Path(sys.argv[4])
shadow_receipt_arg = sys.argv[5]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"missing acceptance artifact: {path}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


baseline_receipt = run_dir / "r9700-baseline-receipt.json"
promotion_path = run_dir / "promotion.json"
benchmark_path = run_dir / "benchmark.json"
calibration_path = run_dir / "calibration.json"
doctor_path = run_dir / "r9700-doctor.json"

promotion = json.loads(
    promotion_path.read_text(encoding="utf-8")
)
evidence = json.loads(
    evidence_path.read_text(encoding="utf-8")
)
scale_readiness = json.loads(
    scale_readiness_path.read_text(
        encoding="utf-8"
    )
)

if evidence.get("git_sha") != expected_sha:
    raise SystemExit(
        "validated evidence SHA does not match requested acceptance SHA"
    )

payload: dict[str, object] = {
    "schema_version": 1,
    "kind": "r9700-exact-sha-acceptance",
    "git_sha": expected_sha,
    "run_dir": str(run_dir),
    "evidence_only": True,
    "runtime_authority_changed": False,
    "dispatch_allowed": False,
    "promotion_passed": bool(
        promotion.get("passed", False)
    ),
    "promotion_failed": promotion.get("failed", []),
    "scale_up_ready": bool(
        scale_readiness.get(
            "passed",
            False,
        )
    ),
    "scale_readiness_failed": (
        scale_readiness.get(
            "failed",
            [],
        )
    ),
    "artifacts": {
        "baseline_receipt": artifact(baseline_receipt),
        "evidence_validation": artifact(evidence_path),
        "scale_readiness": artifact(scale_readiness_path),
        "promotion": artifact(promotion_path),
        "benchmark": artifact(benchmark_path),
        "calibration": artifact(calibration_path),
        "doctor": artifact(doctor_path),
    },
}

if shadow_receipt_arg:
    shadow_receipt = Path(shadow_receipt_arg)
    payload["shadow_evaluation"] = {
        "completed": True,
        "receipt": artifact(shadow_receipt),
    }
else:
    payload["shadow_evaluation"] = {
        "completed": False,
        "reason": "no shadow export was supplied",
    }

print(
    json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )
)
PY

CURRENT_STAGE="complete"
trap - EXIT

echo "R9700 exact-SHA acceptance complete"
echo "run_dir: ${RUN_DIR}"
echo "summary: ${SUMMARY}"
echo "scale_up_ready: ${SCALE_UP_READY}"
