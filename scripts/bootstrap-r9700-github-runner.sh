#!/usr/bin/env bash
set -euo pipefail

REPO="${MY_JEV_GITHUB_REPO:-scottjoyner/my-jev}"
RUNNER_LABEL="${MY_JEV_R9700_RUNNER_LABEL:-r9700}"
RUNNER_NAME="${MY_JEV_R9700_RUNNER_NAME:-$(hostname -s)-my-jev-r9700}"
RUNNER_ROOT="${MY_JEV_R9700_RUNNER_ROOT:-$HOME/actions-runner-my-jev-r9700}"
MIN_GIB="${MY_JEV_R9700_MIN_GIB:-28}"

usage() {
  cat >&2 <<EOF
usage: $0 [--check-only]

Environment overrides:
  MY_JEV_GITHUB_REPO          default: ${REPO}
  MY_JEV_R9700_RUNNER_LABEL  default: ${RUNNER_LABEL}
  MY_JEV_R9700_RUNNER_NAME   default: ${RUNNER_NAME}
  MY_JEV_R9700_RUNNER_ROOT   default: ${RUNNER_ROOT}
  MY_JEV_R9700_MIN_GIB       default: ${MIN_GIB}

The script never prints GitHub registration tokens.
EOF
  exit 64
}

CHECK_ONLY=false
case "${1:-}" in
  "")
    ;;
  --check-only)
    CHECK_ONLY=true
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage
    ;;
esac

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "R9700 runner bootstrap requires Linux" >&2
  exit 65
fi

case "$(uname -m)" in
  x86_64|amd64)
    ARCH="x64"
    ;;
  *)
    echo "unsupported runner architecture: $(uname -m)" >&2
    exit 66
    ;;
esac

command -v git >/dev/null || {
  echo "git is required" >&2
  exit 67
}
command -v gh >/dev/null || {
  echo "GitHub CLI (gh) is required" >&2
  exit 68
}
command -v python >/dev/null || {
  echo "python is required" >&2
  exit 69
}

gh auth status >/dev/null 2>&1 || {
  echo "gh is not authenticated; run gh auth login for the repository owner account" >&2
  exit 70
}

python - "${MIN_GIB}" <<'PY'
from __future__ import annotations

import sys

import torch

minimum_gib = float(sys.argv[1])
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("cuda_available", torch.cuda.is_available())

if not torch.version.hip:
    raise SystemExit("preflight failed: PyTorch is not reporting a HIP runtime")
if not torch.cuda.is_available():
    raise SystemExit("preflight failed: ROCm accelerator is not visible through torch.cuda")

matches = []
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    name = str(props.name)
    gib = float(props.total_memory) / (1024**3)
    print(f"device[{index}]={name} memory_gib={gib:.2f}")
    if "r9700" in name.lower() and gib >= minimum_gib:
        matches.append((index, name, gib))

if not matches:
    raise SystemExit(
        "preflight failed: no visible R9700 device met the minimum "
        f"{minimum_gib:.1f} GiB contract"
    )
PY

RUNNERS_JSON="$(
  gh api     --method GET     -H "Accept: application/vnd.github+json"     "repos/${REPO}/actions/runners?per_page=100"
)"

RUNNER_STATE="$(
  python -     "${RUNNER_NAME}"     "${RUNNER_LABEL}"     <<'PY' <<<"${RUNNERS_JSON}"
import json
import sys

runner_name = sys.argv[1]
required_label = sys.argv[2]
payload = json.load(sys.stdin)

matches = [
    runner
    for runner in payload.get("runners", [])
    if runner.get("name") == runner_name
]

if not matches:
    print("absent")
    raise SystemExit(0)

runner = matches[0]
labels = {
    str(item.get("name"))
    for item in runner.get("labels", [])
}
online = str(runner.get("status")) == "online"
busy = bool(runner.get("busy"))

if required_label not in labels:
    print("wrong-label")
elif not online:
    print("offline")
elif busy:
    print("busy")
else:
    print("ready")
PY
)"

echo "repository: ${REPO}"
echo "runner_name: ${RUNNER_NAME}"
echo "runner_label: ${RUNNER_LABEL}"
echo "runner_state: ${RUNNER_STATE}"

if [[ "${RUNNER_STATE}" == "ready" ]]; then
  echo "R9700 GitHub runner contract is ready"
  exit 0
fi

if ${CHECK_ONLY}; then
  echo "check-only: runner is not ready" >&2
  exit 71
fi

if [[ "${RUNNER_STATE}" == "wrong-label" ]]; then
  echo "existing runner ${RUNNER_NAME} is missing label ${RUNNER_LABEL}" >&2
  echo "refusing to mutate labels automatically; reconfigure that runner or choose a new MY_JEV_R9700_RUNNER_NAME" >&2
  exit 72
fi

if [[ "${RUNNER_STATE}" == "busy" ]]; then
  echo "runner exists and is busy; no bootstrap action required" >&2
  exit 73
fi

if [[ "${RUNNER_STATE}" == "offline" ]]; then
  if [[ -x "${RUNNER_ROOT}/svc.sh" ]]; then
    echo "existing runner is offline; attempting service restart"
    sudo "${RUNNER_ROOT}/svc.sh" start
    sleep 2
  else
    echo "runner is registered but offline and local service files were not found at ${RUNNER_ROOT}" >&2
    exit 74
  fi
else
  mkdir -p "${RUNNER_ROOT}"

  VERSION="$(
    gh api       --method GET       -H "Accept: application/vnd.github+json"       repos/actions/runner/releases/latest       --jq '.tag_name | ltrimstr("v")'
  )"

  TARBALL="actions-runner-linux-${ARCH}-${VERSION}.tar.gz"
  DOWNLOAD_URL="https://github.com/actions/runner/releases/download/v${VERSION}/${TARBALL}"

  echo "installing GitHub Actions runner ${VERSION} into ${RUNNER_ROOT}"
  curl -fL --retry 3     -o "${RUNNER_ROOT}/${TARBALL}"     "${DOWNLOAD_URL}"

  (
    cd "${RUNNER_ROOT}"
    tar xzf "${TARBALL}"
    rm -f "${TARBALL}"

    REGISTRATION_TOKEN="$(
      gh api         --method POST         -H "Accept: application/vnd.github+json"         "repos/${REPO}/actions/runners/registration-token"         --jq .token
    )"

    ./config.sh       --unattended       --replace       --url "https://github.com/${REPO}"       --token "${REGISTRATION_TOKEN}"       --name "${RUNNER_NAME}"       --labels "${RUNNER_LABEL}"       --work "_work"

    unset REGISTRATION_TOKEN

    sudo ./svc.sh install "${USER}"
    sudo ./svc.sh start
  )
fi

for attempt in $(seq 1 15); do
  sleep 2
  STATE="$(
    gh api       --method GET       -H "Accept: application/vnd.github+json"       "repos/${REPO}/actions/runners?per_page=100"       --jq       ".runners[] | select(.name == \"${RUNNER_NAME}\") | [(.status // \"\"), ([.labels[].name] | index(\"${RUNNER_LABEL}\") != null)] | @tsv"       | tail -n 1
  )"

  if [[ "${STATE}" == $'online\ttrue' ]]; then
    echo "R9700 GitHub runner is online with label ${RUNNER_LABEL}"
    exit 0
  fi
done

echo "runner did not become online with the required label" >&2
exit 75
