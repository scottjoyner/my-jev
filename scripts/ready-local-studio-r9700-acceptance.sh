#!/usr/bin/env bash
set -euo pipefail

REPO="${MY_JEV_GITHUB_REPO:-scottjoyner/my-jev}"
WORKFLOW="${MY_JEV_LOCAL_STUDIO_R9700_WORKFLOW:-local-studio-r9700-acceptance.yml}"
RUNNER_LABEL="${MY_JEV_R9700_RUNNER_LABEL:-r9700}"
LOCAL_STUDIO_SHA="${LOCAL_STUDIO_SHA:-4eb7beba8db8c2a380314c983d6292ddf3b6e07e}"
ENV_FILE="${LOCAL_STUDIO_R9700_ACCEPTANCE_ENV_FILE:-$HOME/.config/local-studio/r9700-acceptance.env}"

ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd
)"
cd "${ROOT}"

for command in git gh python; do
  command -v "${command}" >/dev/null || {
    echo "${command} is required" >&2
    exit 64
  }
done

gh auth status >/dev/null 2>&1 || {
  echo "gh is not authenticated; run gh auth login first" >&2
  exit 65
}

if [[ ! "${LOCAL_STUDIO_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "LOCAL_STUDIO_SHA must be exactly 40 lowercase hex characters" >&2
  exit 66
fi

git fetch origin main
MAIN_SHA="$(git rev-parse origin/main)"
HEAD_SHA="$(git rev-parse HEAD)"

if [[ "${HEAD_SHA}" != "${MAIN_SHA}" ]]; then
  echo "my-jev checkout is not exact origin/main" >&2
  echo "HEAD=${HEAD_SHA}" >&2
  echo "origin/main=${MAIN_SHA}" >&2
  echo "Update the checkout before arming the physical acceptance." >&2
  exit 67
fi

if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "my-jev checkout is dirty; refusing to arm a physical acceptance" >&2
  exit 68
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing runner-local acceptance identity file: ${ENV_FILE}" >&2
  echo "Required keys: OPENCODE_SESSION_ID and OPENCODE_EXPECTED_PROVIDER" >&2
  echo "Optional key: OPENCODE_MIN_COMPLETED_TOOLS" >&2
  exit 78
fi

python - "${ENV_FILE}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
if path.stat().st_mode & 0o022:
    raise SystemExit(f"acceptance identity file is group/other writable: {path}")

allowed = {
    "OPENCODE_SESSION_ID",
    "OPENCODE_EXPECTED_PROVIDER",
    "OPENCODE_MIN_COMPLETED_TOOLS",
}
values = {}
for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if "=" not in line:
        raise SystemExit(f"invalid acceptance identity line {number}")
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if key not in allowed:
        raise SystemExit(f"unsupported acceptance identity key: {key}")
    if not value:
        raise SystemExit(f"empty acceptance identity value: {key}")
    values[key] = value

missing = [
    key
    for key in ("OPENCODE_SESSION_ID", "OPENCODE_EXPECTED_PROVIDER")
    if not values.get(key)
]
if missing:
    raise SystemExit("missing required acceptance identity: " + ", ".join(missing))

tool_count = values.get("OPENCODE_MIN_COMPLETED_TOOLS", "1")
if not tool_count.isdigit() or int(tool_count) < 1:
    raise SystemExit("OPENCODE_MIN_COMPLETED_TOOLS must be an integer >= 1")

print("runner-local OpenCode acceptance identity is present and structurally valid")
PY

echo "== Existing Local Studio R9700 workflow =="

ACTIVE_JSON="$(
  gh run list \
    --repo "${REPO}" \
    --workflow "${WORKFLOW}" \
    --limit 30 \
    --json databaseId,status,conclusion,url,event,headBranch,createdAt
)"

CURRENT="$(
  ACTIVE_JSON="${ACTIVE_JSON}" python - <<'PY'
import json
import os

active_states = {"queued", "in_progress", "waiting", "pending"}
runs = json.loads(os.environ["ACTIVE_JSON"])
matches = [
    run
    for run in runs
    if run.get("status") in active_states
    and run.get("event") == "workflow_dispatch"
    and run.get("headBranch") == "main"
]
matches.sort(key=lambda run: run.get("createdAt") or "", reverse=True)
if matches:
    run = matches[0]
    print(
        "\t".join(
            [
                str(run.get("databaseId") or ""),
                str(run.get("status") or ""),
                str(run.get("event") or ""),
                str(run.get("headBranch") or ""),
                str(run.get("url") or ""),
            ]
        )
    )
PY
)"

if [[ -n "${CURRENT}" ]]; then
  echo "== R9700 GitHub runner =="
  MY_JEV_GITHUB_REPO="${REPO}" MY_JEV_R9700_RUNNER_LABEL="${RUNNER_LABEL}" \
    bash scripts/bootstrap-r9700-github-runner.sh

  echo "A current-main physical acceptance already exists:"
  printf '%s\n' "${CURRENT}"
  echo "No duplicate workflow was dispatched."
  exit 0
fi

mapfile -t STALE_RUN_IDS < <(
  ACTIVE_JSON="${ACTIVE_JSON}" python - <<'PY'
import json
import os

active_states = {"queued", "in_progress", "waiting", "pending"}
for run in json.loads(os.environ["ACTIVE_JSON"]):
    if run.get("status") not in active_states:
        continue
    if run.get("event") == "workflow_dispatch" and run.get("headBranch") == "main":
        continue
    run_id = run.get("databaseId")
    if run_id:
        print(run_id)
PY
)

if (( ${#STALE_RUN_IDS[@]} > 0 )); then
  echo "Cancelling stale pre-main physical workflow runs:"
  for run_id in "${STALE_RUN_IDS[@]}"; do
    echo "  run_id=${run_id}"
    gh run cancel "${run_id}" --repo "${REPO}"
  done

  remaining=1
  for _ in $(seq 1 30); do
    remaining=0
    for run_id in "${STALE_RUN_IDS[@]}"; do
      state="$(
        gh run view "${run_id}" --repo "${REPO}" --json status --jq .status 2>/dev/null || true
      )"
      case "${state}" in
        queued|in_progress|waiting|pending)
          remaining=1
          ;;
      esac
    done
    (( remaining == 0 )) && break
    sleep 1
  done

  if (( remaining != 0 )); then
    echo "stale physical workflow runs did not cancel cleanly; refusing duplicate dispatch" >&2
    exit 69
  fi
fi

echo "== R9700 GitHub runner =="
MY_JEV_GITHUB_REPO="${REPO}" MY_JEV_R9700_RUNNER_LABEL="${RUNNER_LABEL}" \
  bash scripts/bootstrap-r9700-github-runner.sh

echo "== Dispatch exact Local Studio candidate =="
gh workflow run "${WORKFLOW}"   --repo "${REPO}"   --ref main   -f "local_studio_sha=${LOCAL_STUDIO_SHA}"   -f "runner_label=${RUNNER_LABEL}"

for _ in $(seq 1 15); do
  sleep 2
  NEW_RUN="$(
    gh run list       --repo "${REPO}"       --workflow "${WORKFLOW}"       --limit 10       --json databaseId,status,conclusion,url,event,headBranch,createdAt       --jq '
        map(select(.event == "workflow_dispatch"))
        | sort_by(.createdAt)
        | reverse
        | .[0] // empty
        | [
            (.databaseId | tostring),
            (.status // ""),
            (.event // ""),
            (.headBranch // ""),
            (.url // "")
          ]
        | @tsv
      '       2>/dev/null || true
  )"
  if [[ -n "${NEW_RUN}" ]]; then
    echo "Physical acceptance dispatched:"
    printf '%s\n' "${NEW_RUN}"
    exit 0
  fi
done

echo "Workflow dispatch was accepted but no matching run appeared within 30 seconds" >&2
exit 70
