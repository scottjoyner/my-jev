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

EVIDENCE_JSON="${RUN_DIR}/r9700-evidence-validation.json"
python -m my_jev.baseline_evidence   --run-dir "${RUN_DIR}"   --expected-sha "${EXPECTED_SHA}"   --output "${EVIDENCE_JSON}"

SHADOW_RECEIPT=""
if [[ -n "${SHADOW_EXPORT}" ]]; then
  if [[ ! -f "${SHADOW_EXPORT}" ]]; then
    echo "shadow export does not exist: ${SHADOW_EXPORT}" >&2
    exit 72
  fi

  SHADOW_LIMIT="${MY_JEV_SHADOW_LIMIT:-200}"
  SHADOW_MIN_PRIORITY="${MY_JEV_SHADOW_MIN_PRIORITY:-0.25}"

  python -m my_jev.shadow_evaluation     --run-dir "${RUN_DIR}"     --expected-sha "${EXPECTED_SHA}"     --shadow-export "${SHADOW_EXPORT}"     --device cuda     --limit "${SHADOW_LIMIT}"     --min-priority "${SHADOW_MIN_PRIORITY}"

  SHADOW_RECEIPT="${RUN_DIR}/shadow-evaluation/shadow-evaluation-receipt.json"
  if [[ ! -f "${SHADOW_RECEIPT}" ]]; then
    echo "shadow evaluation did not produce its receipt" >&2
    exit 73
  fi
fi

SUMMARY="${RUN_DIR}/r9700-acceptance-summary.json"
python -   "${RUN_DIR}"   "${EXPECTED_SHA}"   "${EVIDENCE_JSON}"   "${SHADOW_RECEIPT}"   > "${SUMMARY}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
expected_sha = sys.argv[2]
evidence_path = Path(sys.argv[3])
shadow_receipt_arg = sys.argv[4]


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
    "artifacts": {
        "baseline_receipt": artifact(baseline_receipt),
        "evidence_validation": artifact(evidence_path),
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

echo "R9700 exact-SHA acceptance complete"
echo "run_dir: ${RUN_DIR}"
echo "summary: ${SUMMARY}"
