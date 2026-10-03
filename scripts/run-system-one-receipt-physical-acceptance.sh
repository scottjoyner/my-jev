#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <40-char producer SHA> <40-char auto-assist SHA>" >&2
  exit 64
}

EXPECTED_SHA="${1:-}"
AUTO_ASSIST_SHA="${2:-}"
[[ "${EXPECTED_SHA}" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "${AUTO_ASSIST_SHA}" =~ ^[0-9a-f]{40}$ ]] || usage

ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd
)"
cd "${ROOT}"

if [[ "$(git rev-parse HEAD)" != "${EXPECTED_SHA}" ]]; then
  echo "producer checkout is not at the requested exact SHA" >&2
  exit 65
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
test -x "${PYTHON_BIN}" || {
  echo "preconfigured R9700 Python is unavailable" >&2
  exit 66
}

AUTO_ASSIST_ROOT="${AUTO_ASSIST_ROOT:-}"
test -n "${AUTO_ASSIST_ROOT}" || {
  echo "AUTO_ASSIST_ROOT is required" >&2
  exit 67
}
test -f "${AUTO_ASSIST_ROOT}/src/assistx/system_one_receipt.py" || {
  echo "exact auto-assist receipt consumer is missing" >&2
  exit 67
}

ACTUAL_AUTO_ASSIST_SHA="$(
  git -C "${AUTO_ASSIST_ROOT}" rev-parse HEAD
)"
if [[ "${ACTUAL_AUTO_ASSIST_SHA}" != "${AUTO_ASSIST_SHA}" ]]; then
  echo "auto-assist checkout is not at the requested exact SHA" >&2
  exit 68
fi

CONFIG_FILE="${MY_JEV_RECEIPT_ACCEPTANCE_ENV:-$HOME/.config/my-jev/system-one-receipt-acceptance.env}"
test -f "${CONFIG_FILE}" || {
  echo "runner-local receipt acceptance config is missing" >&2
  exit 69
}

CONFIG_MODE="$(
  stat -c '%a' "${CONFIG_FILE}"
)"
python - "${CONFIG_MODE}" <<'PY'
import sys

mode = int(sys.argv[1], 8)
if mode & 0o022:
    raise SystemExit(
        "runner-local receipt acceptance config must not be group/other writable"
    )
PY

CHECKPOINT=""
CALIBRATION=""
while IFS='=' read -r key value; do
  case "${key}" in
    MY_JEV_RECEIPT_ACCEPTANCE_CHECKPOINT)
      CHECKPOINT="${value}"
      ;;
    MY_JEV_RECEIPT_ACCEPTANCE_CALIBRATION)
      CALIBRATION="${value}"
      ;;
    ""|\#*)
      ;;
  esac
done < "${CONFIG_FILE}"

test -n "${CHECKPOINT}" || {
  echo "MY_JEV_RECEIPT_ACCEPTANCE_CHECKPOINT is required" >&2
  exit 70
}
[[ "${CHECKPOINT}" = /* ]] || {
  echo "acceptance checkpoint path must be absolute" >&2
  exit 70
}

if [[ -d "${CHECKPOINT}" ]]; then
  MODEL_ARTIFACT="${CHECKPOINT}/model.pt"
else
  MODEL_ARTIFACT="${CHECKPOINT}"
fi
test -f "${MODEL_ARTIFACT}" || {
  echo "checkpoint model artifact is missing" >&2
  exit 71
}

if [[ -n "${CALIBRATION}" ]]; then
  [[ "${CALIBRATION}" = /* ]] || {
    echo "configured calibration path must be absolute" >&2
    exit 72
  }
  if [[ ! -f "${CALIBRATION}" ]]; then
    echo "configured calibration file is missing" >&2
    exit 72
  fi
fi

MODEL_ARTIFACT_SHA="$(
  sha256sum "${MODEL_ARTIFACT}" | awk '{print $1}'
)"
MODEL_ID_SHA="$(
  printf '%s' "${CHECKPOINT}" | sha256sum | awk '{print $1}'
)"

PORT="${MY_JEV_RECEIPT_ACCEPTANCE_PORT:-18088}"
if [[ ! "${PORT}" =~ ^[0-9]+$ ]] || (( PORT < 1024 || PORT > 65535 )); then
  echo "acceptance port must be an integer in [1024, 65535]" >&2
  exit 73
fi

"${PYTHON_BIN}" - "${PORT}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", port))
PY

for module in fastapi uvicorn pydantic torch; do
  "${PYTHON_BIN}" - "${module}" <<'PY'
import importlib
import sys

importlib.import_module(sys.argv[1])
PY
done

OUT_ROOT="${RUNNER_TEMP:-/tmp}/system-one-receipt-acceptance-${EXPECTED_SHA}"
rm -rf "${OUT_ROOT}"
mkdir -p "${OUT_ROOT}"

REQUEST_JSON="${OUT_ROOT}/request.json"
RESPONSE_JSON="${OUT_ROOT}/response.json"
SERVER_LOG="${OUT_ROOT}/server.log"
ACCEPTANCE_JSON="${OUT_ROOT}/acceptance.json"

cat > "${REQUEST_JSON}" <<'JSON'
{
  "state": {
    "utterance": "Return a read-only status assessment. Do not take actions.",
    "conversation_summary": "",
    "source": "system-one-physical-receipt-acceptance",
    "speaker_id": "",
    "speaker_verified": false,
    "foreground": true,
    "active_work": [],
    "pending_approvals": [],
    "available_capabilities": ["chat"],
    "available_tools": [],
    "actions_allowed": false,
    "external_actions_allowed": false,
    "privileged_actions_allowed": false,
    "metadata": {
      "acceptance": "physical-receipt"
    }
  },
  "constraints": {
    "speaker_verified": false,
    "actions_allowed": false,
    "local_writes_allowed": false,
    "external_actions_allowed": false,
    "privileged_actions_allowed": false,
    "approval_gate_available": false,
    "active_work": false
  }
}
JSON

SERVER_ARGS=(
  "${PYTHON_BIN}"
  -m my_jev.server
  --checkpoint "${CHECKPOINT}"
  --host 127.0.0.1
  --port "${PORT}"
  --device cuda
)
if [[ -n "${CALIBRATION}" ]]; then
  SERVER_ARGS+=(--calibration "${CALIBRATION}")
fi

PYTHONPATH="${ROOT}/src" "${SERVER_ARGS[@]}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID="$!"

cleanup() {
  local rc="$?"
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        break
      fi
      sleep 0.25
    done
    kill -9 "${SERVER_PID}" 2>/dev/null || true
  fi
  wait "${SERVER_PID}" 2>/dev/null || true

  "${PYTHON_BIN}" - "${PORT}" <<'PY' || rc=74
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    if sock.connect_ex(("127.0.0.1", port)) == 0:
        raise SystemExit("isolated acceptance port remained open after cleanup")
PY
  exit "${rc}"
}
trap cleanup EXIT

READY=false
for _ in $(seq 1 240); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "my-jev acceptance sidecar exited before becoming healthy" >&2
    exit 75
  fi

  if "${PYTHON_BIN}" - "${PORT}" >/dev/null 2>&1 <<'PY'
import json
import sys
import urllib.request

port = int(sys.argv[1])
with urllib.request.urlopen(
    f"http://127.0.0.1:{port}/healthz",
    timeout=1.0,
) as response:
    payload = json.load(response)

if payload.get("ok") is not True:
    raise SystemExit(1)
if payload.get("contract") != "assistx-agent-policy-v1":
    raise SystemExit(1)
PY
  then
    READY=true
    break
  fi
  sleep 1
done

if [[ "${READY}" != "true" ]]; then
  echo "my-jev acceptance sidecar did not become healthy" >&2
  exit 76
fi

"${PYTHON_BIN}" - "${PORT}" "${REQUEST_JSON}" "${RESPONSE_JSON}" <<'PY'
import pathlib
import sys
import urllib.request

port = int(sys.argv[1])
request_path = pathlib.Path(sys.argv[2])
response_path = pathlib.Path(sys.argv[3])

request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/agent-policy",
    data=request_path.read_bytes(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=120.0) as response:
    response_path.write_bytes(response.read())
PY

AUTO_ASSIST_RECEIPT_PATH="${AUTO_ASSIST_ROOT}/src/assistx/system_one_receipt.py"
"${PYTHON_BIN}" - \
  "${AUTO_ASSIST_RECEIPT_PATH}" \
  "${REQUEST_JSON}" \
  "${RESPONSE_JSON}" \
  "${EXPECTED_SHA}" \
  "${AUTO_ASSIST_SHA}" \
  "${CHECKPOINT}" \
  "${MODEL_ID_SHA}" \
  "${MODEL_ARTIFACT_SHA}" \
  "${ACCEPTANCE_JSON}" <<'PY'
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import platform
import sys

(
    receipt_module_path,
    request_path,
    response_path,
    producer_sha,
    auto_assist_sha,
    expected_model_id,
    model_id_sha,
    expected_artifact_sha,
    output_path,
) = sys.argv[1:]

spec = importlib.util.spec_from_file_location(
    "_assistx_system_one_receipt_physical",
    receipt_module_path,
)
if spec is None or spec.loader is None:
    raise SystemExit("could not load exact AssistX receipt consumer")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

request_payload = json.loads(
    pathlib.Path(request_path).read_text(encoding="utf-8")
)
response_payload = json.loads(
    pathlib.Path(response_path).read_text(encoding="utf-8")
)
if response_payload.get("contract") != "assistx-agent-policy-v1":
    raise SystemExit("unexpected physical policy response contract")

receipt = response_payload.get("decision_receipt")
if not isinstance(receipt, dict):
    raise SystemExit("physical policy response is missing decision_receipt")

binding = module.bind_shadow_receipt(
    request_payload=request_payload,
    receipt_value=receipt,
)
provider = binding["provider"]
authority = binding["authority"]

if provider.get("provider_id") != "my-jev":
    raise SystemExit("physical receipt provider_id mismatch")
if provider.get("provider_version") != "assistx-agent-policy-v1":
    raise SystemExit("physical receipt provider_version mismatch")
if provider.get("model_id") != expected_model_id:
    raise SystemExit("physical receipt model_id mismatch")
if provider.get("model_artifact_sha256") != expected_artifact_sha:
    raise SystemExit("physical receipt model artifact SHA mismatch")
runtime = provider.get("runtime")
if not isinstance(runtime, dict) or runtime.get("device") != "cuda":
    raise SystemExit("physical receipt runtime device mismatch")
if binding.get("evidence_only") is not True:
    raise SystemExit("physical receipt ceased to be evidence-only")
if binding.get("authoritative_behavior_changed") is not False:
    raise SystemExit("physical receipt changed authoritative behavior")
if any(value is not False for value in authority.values()):
    raise SystemExit("physical receipt widened authority")

host_sha256 = hashlib.sha256(
    platform.node().encode("utf-8")
).hexdigest()

result = {
    "schema": "my-jev-system-one-physical-receipt-acceptance-v1",
    "status": "pass",
    "host_sha256": host_sha256,
    "producer_sha": producer_sha,
    "auto_assist_sha": auto_assist_sha,
    "request_sha256": binding["request_sha256"],
    "decision_receipt_sha256": binding["decision_receipt_sha256"],
    "model_visible_input_sha256": binding["model_visible_input_sha256"],
    "question_schema_sha256": binding["question_schema_sha256"],
    "candidate_set_sha256": binding["candidate_set_sha256"],
    "response_sha256": binding["response_sha256"],
    "provider_id": provider["provider_id"],
    "provider_version": provider["provider_version"],
    "model_id_sha256": model_id_sha,
    "model_artifact_sha256": expected_artifact_sha,
    "runtime": provider.get("runtime", {}),
    "resolver_result": binding.get("resolver_result"),
    "evidence_only": True,
    "authoritative_behavior_changed": False,
    "authority": authority,
}
pathlib.Path(output_path).write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, indent=2, sort_keys=True))
PY

test -s "${ACCEPTANCE_JSON}"
echo "SYSTEM_ONE_PHYSICAL_RECEIPT_ACCEPTANCE: PASS"
echo "acceptance_json: ${ACCEPTANCE_JSON}"
