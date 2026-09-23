#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <baseline-run-dir> <40-char git SHA>" >&2
  exit 64
}

BASELINE_RUN="${1:-}"
EXPECTED_SHA="${2:-}"
[[ -n "${BASELINE_RUN}" ]] || usage
[[ "${EXPECTED_SHA}" =~ ^[0-9a-f]{40}$ ]] || usage

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

ACTUAL_SHA="$(git rev-parse HEAD)"
if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
  echo "refusing scale-up: HEAD ${ACTUAL_SHA} != expected ${EXPECTED_SHA}" >&2
  exit 65
fi

STATUS="$(git status --porcelain=v1 --untracked-files=all)"
if [[ -n "${STATUS}" ]]; then
  echo "refusing scale-up: checkout is dirty" >&2
  printf '%s\n' "${STATUS}" >&2
  exit 66
fi

BASELINE_RUN="$(cd "${BASELINE_RUN}" && pwd)"
READINESS="${BASELINE_RUN}/r9700-scale-readiness.json"

if [[ ! -f "${READINESS}" ]]; then
  echo "scale-up readiness artifact is missing: ${READINESS}" >&2
  exit 67
fi

python - "${READINESS}" "${EXPECTED_SHA}" <<'PY'
import json
import sys

path = sys.argv[1]
expected_sha = sys.argv[2]
payload = json.load(
    open(
        path,
        encoding="utf-8",
    )
)

if payload.get("git_sha") != expected_sha:
    raise SystemExit(
        "scale-up readiness SHA does not match requested exact SHA"
    )
if not payload.get("passed"):
    failed = payload.get(
        "failed",
        [],
    )
    raise SystemExit(
        "scale-up readiness is HOLD; failed gates: "
        + ", ".join(
            str(item)
            for item in failed
        )
    )
if payload.get("decision") != "scale":
    raise SystemExit(
        "scale-up readiness decision is not 'scale'"
    )

authority = (
    payload.get(
        "authority_boundary"
    )
    or {}
)
if (
    authority.get(
        "runtime_authority_changed"
    )
    is not False
    or authority.get(
        "dispatch_allowed"
    )
    is not False
):
    raise SystemExit(
        "scale-up readiness authority boundary is invalid"
    )

print("baseline scale-readiness gate passed")
PY

SPEC="configs/experiments/assistx-modernbert-r9700-scaleup.toml"
DATA_DIR="runs/datasets/assistx-policy-r9700-scaleup-v1"
PREFLIGHT_DIR="runs/preflight/r9700-scaleup/${EXPECTED_SHA}"
mkdir -p "${PREFLIGHT_DIR}"

python -m my_jev.doctor   --require-gpu   --require-bf16   > "${PREFLIGHT_DIR}/doctor.json"

if [[ ! -f "${DATA_DIR}/prepare_manifest.json" ]]; then
  python -m my_jev.prepare     --output-dir "${DATA_DIR}"     --records 50000     --seed 23     > "${PREFLIGHT_DIR}/prepare.json"
else
  python - "${DATA_DIR}/prepare_manifest.json" <<'PY'
import json
import sys

from my_jev.agentic_synth import (
    GENERATOR_VERSION,
)

manifest = json.load(
    open(
        sys.argv[1],
        encoding="utf-8",
    )
)
expected = {
    "records_requested": 50000,
    "records_emitted": 50000,
    "seed": 23,
    "generator": GENERATOR_VERSION,
}
wrong = {
    key: {
        "actual": manifest.get(key),
        "expected": value,
    }
    for key, value in expected.items()
    if manifest.get(key) != value
}
if wrong:
    raise SystemExit(
        "existing scale-up dataset violates pinned contract: "
        + repr(wrong)
    )

for split in (
    "train",
    "validation",
    "calibration",
    "test",
):
    item = (
        manifest.get(
            "splits"
        )
        or {}
    ).get(split)
    if (
        not item
        or not item.get(
            "sha256"
        )
        or not item.get(
            "records"
        )
    ):
        raise SystemExit(
            f"scale-up dataset manifest is incomplete for {split}"
        )

print(
    "existing deterministic 50k scale-up dataset contract verified"
)
PY
fi

python -m my_jev.experiment   "${SPEC}"   --dry-run   | tee "${PREFLIGHT_DIR}/dry-run.log"

RESULT_JSON="${PREFLIGHT_DIR}/result.json"
python -m my_jev.experiment   "${SPEC}"   | tee "${RESULT_JSON}"

RUN_DIR="$(
  python - "${RESULT_JSON}" <<'PY'
import json
import sys

payload = json.load(
    open(
        sys.argv[1],
        encoding="utf-8",
    )
)
print(
    payload["run_dir"]
)
PY
)"

python -   "${BASELINE_RUN}"   "${READINESS}"   "${RUN_DIR}"   "${EXPECTED_SHA}"   > "${RUN_DIR}/r9700-scaleup-receipt.json" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

baseline_run = Path(sys.argv[1])
readiness_path = Path(sys.argv[2])
run_dir = Path(sys.argv[3])
expected_sha = sys.argv[4]


def load_json(
    path: Path,
) -> dict:
    if not path.is_file():
        raise SystemExit(
            f"missing required scale-up artifact: {path}"
        )
    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(
        value,
        dict,
    ):
        raise SystemExit(
            f"scale-up artifact is not a JSON object: {path}"
        )
    return value


def sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()
    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )
    return digest.hexdigest()


manifest_path = (
    run_dir
    / "manifest.json"
)
benchmark_path = (
    run_dir
    / "benchmark.json"
)
promotion_path = (
    run_dir
    / "promotion.json"
)
calibration_path = (
    run_dir
    / "calibration.json"
)
checkpoint_config_path = (
    run_dir
    / "checkpoints"
    / "best"
    / "my_jev_config.json"
)

manifest = load_json(
    manifest_path
)
benchmark = load_json(
    benchmark_path
)
promotion = load_json(
    promotion_path
)
load_json(
    calibration_path
)
load_json(
    checkpoint_config_path
)
readiness = load_json(
    readiness_path
)

git_state = (
    manifest.get(
        "git"
    )
    or {}
)
if (
    git_state.get(
        "revision"
    )
    != expected_sha
):
    raise SystemExit(
        "scale-up manifest SHA mismatch"
    )
if (
    git_state.get(
        "dirty"
    )
    is not False
):
    raise SystemExit(
        "scale-up manifest records a dirty checkout"
    )
if (
    readiness.get(
        "git_sha"
    )
    != expected_sha
    or not readiness.get(
        "passed"
    )
):
    raise SystemExit(
        "parent scale-readiness evidence is invalid"
    )

artifacts = {}
for name, path in (
    (
        "manifest",
        manifest_path,
    ),
    (
        "benchmark",
        benchmark_path,
    ),
    (
        "promotion",
        promotion_path,
    ),
    (
        "calibration",
        calibration_path,
    ),
    (
        "checkpoint_config",
        checkpoint_config_path,
    ),
    (
        "parent_scale_readiness",
        readiness_path,
    ),
):
    artifacts[name] = {
        "path": str(
            path
        ),
        "sha256": sha256(
            path
        ),
        "bytes": (
            path.stat().st_size
        ),
    }

payload = {
    "schema_version": 1,
    "kind": (
        "r9700-modernbert-scaleup"
    ),
    "git_sha": expected_sha,
    "baseline_run_dir": str(
        baseline_run
    ),
    "scaleup_run_dir": str(
        run_dir
    ),
    "parent_readiness_decision": (
        readiness.get(
            "decision"
        )
    ),
    "promotion_passed": bool(
        promotion.get(
            "passed",
            False,
        )
    ),
    "promotion_failed": (
        promotion.get(
            "failed",
            []
        )
    ),
    "benchmark_summary": {
        "accuracy": (
            benchmark.get(
                "normal",
                {},
            ).get(
                "accuracy"
            )
        ),
        "ece": (
            benchmark.get(
                "normal",
                {},
            ).get(
                "ece"
            )
        ),
        "accuracy_delta_vs_shuffled": (
            benchmark.get(
                "controls",
                {},
            ).get(
                "accuracy_delta_vs_shuffled"
            )
        ),
        "decisions_per_second": (
            benchmark.get(
                "latency",
                {},
            ).get(
                "decisions_per_second"
            )
        ),
    },
    "evidence_only": True,
    "runtime_authority_changed": False,
    "dispatch_allowed": False,
    "artifacts": artifacts,
}

print(
    json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )
)
PY

echo "R9700 50k scale-up complete"
echo "run_dir: ${RUN_DIR}"
echo "receipt: ${RUN_DIR}/r9700-scaleup-receipt.json"
