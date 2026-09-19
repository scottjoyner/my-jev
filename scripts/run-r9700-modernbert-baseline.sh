#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <40-char git SHA>" >&2
  echo "Runs the small full-encoder ModernBERT baseline only from that exact clean checkout." >&2
  exit 64
}

EXPECTED_SHA="${1:-}"
[[ "${EXPECTED_SHA}" =~ ^[0-9a-f]{40}$ ]] || usage

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

ACTUAL_SHA="$(git rev-parse HEAD)"
if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
  echo "refusing baseline: HEAD ${ACTUAL_SHA} != expected ${EXPECTED_SHA}" >&2
  exit 65
fi

STATUS="$(git status --porcelain=v1 --untracked-files=all)"
if [[ -n "${STATUS}" ]]; then
  echo "refusing baseline: checkout is dirty" >&2
  printf '%s\n' "${STATUS}" >&2
  exit 66
fi

MODULE_PATH="$(python - <<'PY'
from pathlib import Path
import my_jev

print(Path(my_jev.__file__).resolve())
PY
)"
EXPECTED_MODULE="${ROOT}/src/my_jev/__init__.py"
if [[ "${MODULE_PATH}" != "${EXPECTED_MODULE}" ]]; then
  echo "refusing baseline: imported my_jev is not this checkout" >&2
  echo "resolved: ${MODULE_PATH}" >&2
  echo "expected: ${EXPECTED_MODULE}" >&2
  echo "install this checkout with: python -m pip install -e '.[dev]'" >&2
  exit 67
fi

SPEC="configs/experiments/assistx-modernbert-r9700-baseline.toml"
DATA_DIR="runs/datasets/assistx-policy-r9700-baseline-v1"
PREFLIGHT_DIR="runs/preflight/r9700/${EXPECTED_SHA}"
mkdir -p "${PREFLIGHT_DIR}"

DOCTOR_JSON="${PREFLIGHT_DIR}/doctor.json"
python -m my_jev.doctor --require-gpu --require-bf16 > "${DOCTOR_JSON}"
cat "${DOCTOR_JSON}"

python - "${DOCTOR_JSON}" <<'PY'
import json
import os
import sys

path = sys.argv[1]
report = json.load(open(path, encoding="utf-8"))

if not report.get("hip_version"):
    raise SystemExit("R9700 preflight failed: PyTorch is not reporting a HIP runtime")

pattern = os.environ.get("MY_JEV_R9700_DEVICE_PATTERN", "R9700").lower()
minimum_gib = float(os.environ.get("MY_JEV_R9700_MIN_GIB", "28"))
devices = report.get("devices") or []

matches = [
    device
    for device in devices
    if pattern in str(device.get("name", "")).lower()
    and float(device.get("total_memory_gib", 0.0)) >= minimum_gib
]
if not matches:
    available = ", ".join(
        f"{device.get('name')} ({float(device.get('total_memory_gib', 0.0)):.1f} GiB)"
        for device in devices
    ) or "none"
    raise SystemExit(
        "R9700 preflight failed: no visible device matched "
        f"pattern={pattern!r} with >= {minimum_gib:.1f} GiB; visible: {available}"
    )

print(
    "R9700 preflight matched: "
    + ", ".join(
        f"{device.get('name')} ({float(device.get('total_memory_gib', 0.0)):.1f} GiB)"
        for device in matches
    )
)
PY

if [[ ! -f "${DATA_DIR}/prepare_manifest.json" ]]; then
  python -m my_jev.prepare \
    --output-dir "${DATA_DIR}" \
    --records 2048 \
    --seed 23
else
  python - "${DATA_DIR}/prepare_manifest.json" <<'PY'
import json
import sys

from my_jev.agentic_synth import GENERATOR_VERSION

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "records_requested": 2048,
    "records_emitted": 2048,
    "seed": 23,
    "generator": GENERATOR_VERSION,
}
wrong = {
    key: (manifest.get(key), value)
    for key, value in expected.items()
    if manifest.get(key) != value
}
if wrong:
    raise SystemExit(
        "existing R9700 baseline dataset does not match the pinned contract: "
        + repr(wrong)
    )

for split in ("train", "validation", "calibration", "test"):
    item = (manifest.get("splits") or {}).get(split)
    if not item or not item.get("sha256") or not item.get("records"):
        raise SystemExit(f"existing dataset manifest is incomplete for {split}")

print("existing deterministic R9700 baseline dataset contract verified")
PY
fi

DRY_RUN_LOG="${PREFLIGHT_DIR}/dry-run.log"
python -m my_jev.experiment "${SPEC}" --dry-run | tee "${DRY_RUN_LOG}"

RESULT_JSON="${PREFLIGHT_DIR}/result.json"
python -m my_jev.experiment "${SPEC}" | tee "${RESULT_JSON}"

RUN_DIR="$(python - "${RESULT_JSON}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["run_dir"])
PY
)"

cp "${DOCTOR_JSON}" "${RUN_DIR}/r9700-doctor.json"
python -m pip freeze > "${RUN_DIR}/pip-freeze.txt"

python - "${RUN_DIR}" "${EXPECTED_SHA}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
expected_sha = sys.argv[2]


def load_json(relative: str):
    path = run_dir / relative
    if not path.is_file():
        raise SystemExit(f"missing required baseline artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


manifest = load_json("manifest.json")
git_state = manifest.get("git") or {}
if git_state.get("revision") != expected_sha:
    raise SystemExit(
        "experiment manifest git revision mismatch: "
        f"{git_state.get('revision')} != {expected_sha}"
    )
if git_state.get("dirty") is not False:
    raise SystemExit(
        "experiment manifest did not record a clean checkout: "
        f"{git_state.get('status')!r}"
    )

checkpoint = load_json("checkpoints/best/my_jev_config.json")
run_config = ((checkpoint.get("extra") or {}).get("run_config") or {})
required_config = {
    "device": "cuda",
    "epochs": 1,
    "batch_size": 2,
    "grad_accum": 8,
    "max_state_length": 2048,
    "max_candidate_length": 192,
    "bf16_requested": True,
    "bf16_enabled": True,
    "freeze_backbone": False,
    "gradient_checkpointing": True,
}
wrong = {
    key: (run_config.get(key), value)
    for key, value in required_config.items()
    if run_config.get(key) != value
}
if wrong:
    raise SystemExit("checkpoint run_config failed the R9700 baseline contract: " + repr(wrong))

for relative in ("calibration.json", "benchmark.json", "promotion.json", "r9700-doctor.json"):
    load_json(relative)

required_files = [
    "manifest.json",
    "checkpoints/best/model.pt",
    "checkpoints/best/my_jev_config.json",
    "calibration.json",
    "benchmark.json",
    "promotion.json",
    "r9700-doctor.json",
    "pip-freeze.txt",
]
artifacts = {}
for relative in required_files:
    path = run_dir / relative
    if not path.is_file():
        raise SystemExit(f"missing required baseline artifact: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    artifacts[relative] = {
        "sha256": digest,
        "bytes": path.stat().st_size,
    }

receipt = {
    "schema_version": 1,
    "kind": "r9700-modernbert-baseline",
    "git_sha": expected_sha,
    "run_id": manifest.get("run_id"),
    "spec_sha256": manifest.get("spec_sha256"),
    "dataset_sha256": {
        name: item.get("sha256")
        for name, item in (manifest.get("datasets") or {}).items()
    },
    "artifacts": artifacts,
}
receipt_path = run_dir / "r9700-baseline-receipt.json"
receipt_path.write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(receipt, indent=2, sort_keys=True))
PY

echo "R9700 baseline artifact set validated: ${RUN_DIR}"
echo "receipt: ${RUN_DIR}/r9700-baseline-receipt.json"
