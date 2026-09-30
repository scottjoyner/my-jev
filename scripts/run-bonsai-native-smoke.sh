#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  run-bonsai-native-smoke.sh <PrismML-Eng/llama.cpp checkout> <Bonsai2.gguf> [DecisionRecord.jsonl] [output-dir]

If no DecisionRecord is supplied, a deterministic synthetic AssistX policy record
is generated for the smoke pass.
EOF
  exit 64
}

LLAMA_ROOT="${1:-}"
MODEL="${2:-}"
RECORD="${3:-}"
OUTPUT_DIR="${4:-}"

[[ -n "${LLAMA_ROOT}" && -n "${MODEL}" ]] || usage

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

LLAMA_ROOT="$(cd "${LLAMA_ROOT}" && pwd)"
MODEL="$(cd "$(dirname "${MODEL}")" && pwd)/$(basename "${MODEL}")"

[[ -f "${MODEL}" ]] || {
  echo "Bonsai GGUF does not exist: ${MODEL}" >&2
  exit 65
}

RUNTIME_SHA="$(git -C "${LLAMA_ROOT}" rev-parse HEAD)"
[[ "${RUNTIME_SHA}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "could not resolve exact PrismML llama.cpp SHA" >&2
  exit 66
}

if [[ -z "${OUTPUT_DIR}" ]]; then
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  OUTPUT_DIR="runs/bonsai2/native-smoke/${STAMP}-${RUNTIME_SHA:0:8}"
fi

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "== Probe PrismML Bonsai runtime contract =="
python -m my_jev.bonsai_probe   --llama-root "${LLAMA_ROOT}"   --model-path "${MODEL}"   --output "${OUTPUT_DIR}/runtime-probe.json"

echo "== Build native activation bridge =="
BUILD_DIR="${OUTPUT_DIR}/build"
bash scripts/build-bonsai-bridge.sh   "${LLAMA_ROOT}"   "${BUILD_DIR}"

BRIDGE="${BUILD_DIR}/bin/my-jev-bonsai-bridge"
if [[ ! -x "${BRIDGE}" ]]; then
  BRIDGE="$(
    find "${BUILD_DIR}" -type f -name my-jev-bonsai-bridge -perm -111 | head -n 1
  )"
fi
[[ -n "${BRIDGE}" && -x "${BRIDGE}" ]] || {
  echo "native Bonsai bridge executable was not found" >&2
  exit 67
}

if [[ -z "${RECORD}" ]]; then
  RECORD="${OUTPUT_DIR}/smoke-record.jsonl"
  python - "${RECORD}" <<'PY'
import sys

from my_jev.agentic_synth import (
    generate_agent_policy_records,
)

record = generate_agent_policy_records(
    1,
    seed=137,
)[0]

with open(
    sys.argv[1],
    "w",
    encoding="utf-8",
) as handle:
    handle.write(
        record.model_dump_json()
        + "\n"
    )
PY
else
  RECORD="$(cd "$(dirname "${RECORD}")" && pwd)/$(basename "${RECORD}")"
  [[ -f "${RECORD}" ]] || {
    echo "DecisionRecord does not exist: ${RECORD}" >&2
    exit 68
  }
fi

echo "== Capture exact Bonsai hidden states =="
FRAME="${OUTPUT_DIR}/activation-frame.bin"
CAPTURE_RECEIPT="${OUTPUT_DIR}/activation-capture-receipt.json"

python -m my_jev.bonsai_native   --bridge "${BRIDGE}"   --model "${MODEL}"   --llama-root "${LLAMA_ROOT}"   --record "${RECORD}"   --output "${FRAME}"   --receipt "${CAPTURE_RECEIPT}"   --ctx "${MY_JEV_BONSAI_CTX:-2048}"   --gpu-layers "${MY_JEV_BONSAI_GPU_LAYERS:-999}"

echo "== Run decision-attention head smoke =="
HEAD_RECEIPT="${OUTPUT_DIR}/decision-head-smoke.json"
python -m my_jev.bonsai_smoke   --frame "${FRAME}"   --output "${HEAD_RECEIPT}"   --rank 256   --seed 113

echo "== Bind end-to-end smoke evidence =="
python -   "${OUTPUT_DIR}/runtime-probe.json"   "${CAPTURE_RECEIPT}"   "${HEAD_RECEIPT}"   "${FRAME}"   > "${OUTPUT_DIR}/bonsai-native-smoke-receipt.json" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

probe_path = Path(sys.argv[1])
capture_path = Path(sys.argv[2])
head_path = Path(sys.argv[3])
frame_path = Path(sys.argv[4])


def load(path: Path) -> dict:
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(
        payload,
        dict,
    ):
        raise SystemExit(
            f"not a JSON object: {path}"
        )
    return payload


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


probe = load(probe_path)
capture = load(capture_path)
head = load(head_path)

if not probe.get(
    "ready_for_activation_bridge"
):
    raise SystemExit(
        "runtime probe is not bridge-ready"
    )

capture_model = (
    capture.get("model")
    or {}
)
capture_runtime = (
    capture.get("runtime")
    or {}
)
head_frame = (
    head.get("frame")
    or {}
)

if (
    capture_model.get("sha256")
    != head_frame.get(
        "model_sha256"
    )
):
    raise SystemExit(
        "model SHA differs between capture and head smoke"
    )

if (
    capture_runtime.get("git_sha")
    != head_frame.get(
        "runtime_revision"
    )
):
    raise SystemExit(
        "runtime SHA differs between capture and head smoke"
    )

authority = (
    head.get(
        "authority_boundary"
    )
    or {}
)
if (
    authority.get(
        "dispatch_allowed"
    )
    is not False
    or authority.get(
        "runtime_authority_changed"
    )
    is not False
):
    raise SystemExit(
        "head smoke violated authority boundary"
    )

payload = {
    "schema_version": 1,
    "kind": (
        "bonsai2-native-end-to-end-smoke"
    ),
    "passed": True,
    "model_sha256": (
        capture_model.get(
            "sha256"
        )
    ),
    "runtime_git_sha": (
        capture_runtime.get(
            "git_sha"
        )
    ),
    "frame_sha256": sha256(
        frame_path
    ),
    "state_tokens": (
        head_frame.get(
            "state_tokens"
        )
    ),
    "options": (
        head_frame.get(
            "options"
        )
    ),
    "questions": (
        head_frame.get(
            "questions"
        )
    ),
    "hidden_size": (
        head_frame.get(
            "hidden_size"
        )
    ),
    "tap_count": (
        head_frame.get(
            "tap_count"
        )
    ),
    "head_trained": False,
    "quality_validated": False,
    "authority_boundary": {
        "evidence_only": True,
        "dispatch_allowed": False,
        "runtime_authority_changed": False,
    },
    "artifacts": {
        "probe": str(
            probe_path
        ),
        "capture": str(
            capture_path
        ),
        "head_smoke": str(
            head_path
        ),
        "frame": str(
            frame_path
        ),
    },
}

print(
    json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )
)
PY

echo "Bonsai native end-to-end smoke complete"
echo "output_dir: ${OUTPUT_DIR}"
echo "receipt: ${OUTPUT_DIR}/bonsai-native-smoke-receipt.json"
