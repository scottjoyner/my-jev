#!/usr/bin/env bash
set -euo pipefail

REPO="${MY_JEV_GITHUB_REPO:-scottjoyner/my-jev}"
RUNNER_LABEL="${MY_JEV_R9700_RUNNER_LABEL:-r9700}"
RUNNER_NAME="${MY_JEV_R9700_RUNNER_NAME:-$(hostname -s)-my-jev-r9700}"
RUNNER_ROOT="${MY_JEV_R9700_RUNNER_ROOT:-$HOME/actions-runner-my-jev-r9700}"
MIN_GIB="${MY_JEV_R9700_MIN_GIB:-28}"
DEVICE_PATTERN="${MY_JEV_R9700_DEVICE_PATTERN:-R9700}"

usage() {
  cat >&2 <<EOF
usage: $0 [--check-only]

Environment overrides:
  MY_JEV_GITHUB_REPO          default: ${REPO}
  MY_JEV_R9700_RUNNER_LABEL  default: ${RUNNER_LABEL}
  MY_JEV_R9700_RUNNER_NAME   default: ${RUNNER_NAME}
  MY_JEV_R9700_RUNNER_ROOT   default: ${RUNNER_ROOT}
  MY_JEV_R9700_MIN_GIB       default: ${MIN_GIB}
  MY_JEV_R9700_DEVICE_PATTERN default: ${DEVICE_PATTERN}

This script never prints GitHub registration tokens.
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

for command in gh python; do
  command -v "${command}" >/dev/null || {
    echo "${command} is required" >&2
    exit 67
  }
done

gh auth status >/dev/null 2>&1 || {
  echo "gh is not authenticated; run gh auth login first" >&2
  exit 68
}

python - "${MIN_GIB}" "${DEVICE_PATTERN}" <<'PY'
from __future__ import annotations

import sys

import torch

minimum_gib = float(sys.argv[1])
device_pattern = sys.argv[2].lower()

print("torch", torch.__version__)
print("hip", torch.version.hip)
print("cuda_available", torch.cuda.is_available())

if not torch.version.hip:
    raise SystemExit(
        "preflight failed: PyTorch is not reporting a HIP runtime"
    )
if not torch.cuda.is_available():
    raise SystemExit(
        "preflight failed: ROCm accelerator is not visible through torch.cuda"
    )

matches = []
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    name = str(props.name)
    gib = float(props.total_memory) / (1024**3)
    print(
        f"device[{index}]={name} memory_gib={gib:.2f}"
    )
    if (
        device_pattern in name.lower()
        and gib >= minimum_gib
    ):
        matches.append(
            (index, name, gib)
        )

if not matches:
    raise SystemExit(
        "preflight failed: no visible accelerator matched "
        f"{device_pattern!r} with >= {minimum_gib:.1f} GiB"
    )
PY

RUNNERS_JSON="$(
  gh api     --method GET     -H "Accept: application/vnd.github+json"     "repos/${REPO}/actions/runners?per_page=100"
)"

RUNNER_STATE="$(
  RUNNERS_JSON="${RUNNERS_JSON}"   python -     "${RUNNER_NAME}"     "${RUNNER_LABEL}"     <<'PY'
import json
import os
import sys

runner_name = sys.argv[1]
required_label = sys.argv[2]
payload = json.loads(
    os.environ["RUNNERS_JSON"]
)

matches = [
    runner
    for runner in payload.get(
        "runners",
        [],
    )
    if runner.get("name") == runner_name
]

if not matches:
    print("absent")
    raise SystemExit(0)

runner = matches[0]
labels = {
    str(item.get("name"))
    for item in runner.get(
        "labels",
        [],
    )
}
online = (
    str(runner.get("status"))
    == "online"
)
busy = bool(
    runner.get("busy")
)

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
  echo "existing runner is missing label ${RUNNER_LABEL}" >&2
  echo "choose a new runner name or reconfigure the existing runner" >&2
  exit 72
fi

if [[ "${RUNNER_STATE}" == "busy" ]]; then
  echo "runner is online with the required label and currently busy"
  echo "it is ready to accept the baseline once the current job releases it"
  exit 0
fi

if [[ "${RUNNER_STATE}" == "offline" ]]; then
  command -v sudo >/dev/null || {
    echo "sudo is required to restart the offline runner service" >&2
    exit 74
  }

  if [[ ! -x "${RUNNER_ROOT}/svc.sh" ]]; then
    echo "registered runner is offline but svc.sh is absent at ${RUNNER_ROOT}" >&2
    exit 74
  fi

  (
    cd "${RUNNER_ROOT}"
    sudo ./svc.sh start
  )
else
  mkdir -p "${RUNNER_ROOT}"

  RELEASE_JSON="$(
    gh api       --method GET       -H "Accept: application/vnd.github+json"       repos/actions/runner/releases/latest
  )"

  VERSION="$(
    RELEASE_JSON="${RELEASE_JSON}" python - <<'PY'
import json
import os

payload = json.loads(
    os.environ["RELEASE_JSON"]
)
tag = str(
    payload["tag_name"]
)
print(
    tag[1:]
    if tag.startswith("v")
    else tag
)
PY
  )"

  TARBALL="actions-runner-linux-${ARCH}-${VERSION}.tar.gz"

  ASSET_JSON="$(
    RELEASE_JSON="${RELEASE_JSON}" TARBALL="${TARBALL}" python - <<'PY'
import json
import os

payload = json.loads(
    os.environ["RELEASE_JSON"]
)
tarball = os.environ["TARBALL"]

for asset in payload.get(
    "assets",
    [],
):
    if asset.get("name") == tarball:
        print(
            json.dumps(
                {
                    "url": asset.get(
                        "browser_download_url"
                    ),
                    "digest": asset.get(
                        "digest"
                    ),
                }
            )
        )
        break
else:
    raise SystemExit(
        f"release asset not found: {tarball}"
    )
PY
  )"

  DOWNLOAD_URL="$(
    ASSET_JSON="${ASSET_JSON}" python - <<'PY'
import json
import os

payload = json.loads(
    os.environ["ASSET_JSON"]
)
print(
    payload["url"]
)
PY
  )"

  ASSET_DIGEST="$(
    ASSET_JSON="${ASSET_JSON}" python - <<'PY'
import json
import os

payload = json.loads(
    os.environ["ASSET_JSON"]
)
print(
    payload.get("digest")
    or ""
)
PY
  )"

  echo "installing GitHub Actions runner ${VERSION} into ${RUNNER_ROOT}"

  curl -fL --retry 3 --retry-delay 2 \
    -o "${RUNNER_ROOT}/${TARBALL}" \
    "${DOWNLOAD_URL}"

  if [[ "${ASSET_DIGEST}" == sha256:* ]]; then
    EXPECTED_DIGEST="${ASSET_DIGEST#sha256:}"
    printf '%s  %s\n' \
      "${EXPECTED_DIGEST}" \
      "${RUNNER_ROOT}/${TARBALL}" \
      | sha256sum -c -
  else
    echo "GitHub release asset did not publish a SHA-256 digest; TLS download only" >&2
  fi

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

for _ in $(seq 1 15); do
  sleep 2

  STATE="$(
    gh api       --method GET       -H "Accept: application/vnd.github+json"       "repos/${REPO}/actions/runners?per_page=100"       --jq       ".runners[] | select(.name == \"${RUNNER_NAME}\") | [(.status // \"\"), ([.labels[].name] | index(\"${RUNNER_LABEL}\") != null)] | @tsv"       | tail -n 1
  )"

  if [[ "${STATE}" == $'online\ttrue' ]]; then
    echo "R9700 GitHub runner is online with label ${RUNNER_LABEL}"
    exit 0
  fi
done

echo "runner did not become online with required label ${RUNNER_LABEL}" >&2
exit 75
