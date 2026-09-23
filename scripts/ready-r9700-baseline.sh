#!/usr/bin/env bash
set -euo pipefail

REPO="${MY_JEV_GITHUB_REPO:-scottjoyner/my-jev}"
PR_NUMBER="${MY_JEV_PR_NUMBER:-1}"
PR_BRANCH="${MY_JEV_PR_BRANCH:-feature/system-one-v0}"
RUNNER_LABEL="${MY_JEV_R9700_RUNNER_LABEL:-r9700}"
WORKTREE_ROOT="${MY_JEV_R9700_WORKTREE_ROOT:-$HOME/my-jev-r9700-worktrees}"
SHADOW_EXPORT="${1:-}"

ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd
)"
cd "${ROOT}"

for command in git python cut; do
  command -v "${command}" >/dev/null || {
    echo "${command} is required" >&2
    exit 64
  }
done

if [[ -n "${SHADOW_EXPORT}" && ! -f "${SHADOW_EXPORT}" ]]; then
  echo "shadow export does not exist: ${SHADOW_EXPORT}" >&2
  exit 67
fi

echo "== Local R9700 readiness =="
bash scripts/preflight-r9700-host.sh

GH_AVAILABLE=false
if command -v gh >/dev/null 2>&1   && gh auth status >/dev/null 2>&1; then
  GH_AVAILABLE=true
fi

resolve_exact_sha() {
  if ${GH_AVAILABLE}; then
    local sha
    sha="$(
      gh pr view "${PR_NUMBER}"         --repo "${REPO}"         --json headRefOid         --jq .headRefOid         2>/dev/null || true
    )"
    if [[ "${sha}" =~ ^[0-9a-f]{40}$ ]]; then
      printf '%s\n' "${sha}"
      return 0
    fi
  fi

  git fetch origin "${PR_BRANCH}"
  git rev-parse FETCH_HEAD
}

EXACT_SHA="$(resolve_exact_sha)"

if [[ ! "${EXACT_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "could not resolve exact PR/branch head SHA" >&2
  exit 66
fi

echo "repository: ${REPO}"
echo "pr: #${PR_NUMBER}"
echo "branch: ${PR_BRANCH}"
echo "exact_sha: ${EXACT_SHA}"
echo "rocm_python: $(command -v python)"

find_matching_run() {
  local exact_sha="$1"

  if ! ${GH_AVAILABLE}; then
    return 0
  fi

  gh run list     --repo "${REPO}"     --limit 50     --json databaseId,headSha,name,status,conclusion,url     2>/dev/null     | EXACT_SHA="${exact_sha}" python -c '
import json
import os
import sys

exact_sha = os.environ["EXACT_SHA"]

try:
    runs = json.load(sys.stdin)
except Exception:
    raise SystemExit(0)

for run in runs:
    if (
        run.get("name")
        == "r9700-baseline"
        and run.get("headSha")
        == exact_sha
    ):
        fields = [
            str(
                run.get(
                    "databaseId"
                )
                or ""
            ),
            str(
                run.get(
                    "status"
                )
                or ""
            ),
            str(
                run.get(
                    "conclusion"
                )
                or ""
            ),
            str(
                run.get(
                    "url"
                )
                or ""
            ),
        ]
        print(
            "\t".join(
                fields
            )
        )
        break
'
}

try_actions_path() {
  if ! ${GH_AVAILABLE}; then
    echo "GitHub CLI is unavailable or unauthenticated; using local exact-SHA path"
    return 1
  fi

  echo "== Optional GitHub Actions path =="

  if ! MY_JEV_GITHUB_REPO="${REPO}"     MY_JEV_R9700_RUNNER_LABEL="${RUNNER_LABEL}"     bash scripts/bootstrap-r9700-github-runner.sh; then
    echo "self-hosted runner bootstrap/control is unavailable; using local exact-SHA path" >&2
    return 1
  fi

  local rocm_python
  rocm_python="$(command -v python)"

  if ! gh variable set MY_JEV_R9700_PYTHON     --repo "${REPO}"     --body "${rocm_python}"; then
    echo "cannot set Actions ROCm Python variable; using local exact-SHA path" >&2
    return 1
  fi

  if [[ "${RUNNER_LABEL}" != "r9700" ]]; then
    if ! gh variable set MY_JEV_R9700_RUNNER_LABEL       --repo "${REPO}"       --body "${RUNNER_LABEL}"; then
      echo "cannot set runner-label variable; using local exact-SHA path" >&2
      return 1
    fi
  fi

  if [[ -n "${SHADOW_EXPORT}" ]]; then
    if ! gh variable set MY_JEV_ASSISTX_SHADOW_EXPORT       --repo "${REPO}"       --body "${SHADOW_EXPORT}"; then
      echo "cannot set shadow-export variable; using local exact-SHA path" >&2
      return 1
    fi
  fi

  local existing_run
  existing_run="$(
    find_matching_run "${EXACT_SHA}"
  )"

  if [[ -n "${existing_run}" ]]; then
    local existing_status
    local existing_conclusion

    existing_status="$(
      printf '%s\n' "${existing_run}"         | cut -f2
    )"
    existing_conclusion="$(
      printf '%s\n' "${existing_run}"         | cut -f3
    )"

    if [[ "${existing_status}" != "completed"       || "${existing_conclusion}" == "success" ]]; then
      echo "matching R9700 workflow already exists:"
      printf '%s\n' "${existing_run}"
      return 0
    fi

    echo "previous exact-SHA R9700 workflow did not succeed:"
    printf '%s\n' "${existing_run}"
    echo "re-arming the guarded workflow"
  fi

  if ! gh label create run-r9700     --repo "${REPO}"     --description "Explicitly run exact-SHA R9700 baseline acceptance"     --color B60205     --force; then
    echo "cannot create/manage run-r9700 label; using local exact-SHA path" >&2
    return 1
  fi

  local current_sha
  current_sha="$(
    gh pr view "${PR_NUMBER}"       --repo "${REPO}"       --json headRefOid       --jq .headRefOid       2>/dev/null || true
  )"
  if [[ "${current_sha}" =~ ^[0-9a-f]{40}$     && "${current_sha}" != "${EXACT_SHA}" ]]; then
    echo "PR head moved during setup: ${EXACT_SHA} -> ${current_sha}"
    echo "using new head"
    EXACT_SHA="${current_sha}"
  fi

  gh pr edit "${PR_NUMBER}"     --repo "${REPO}"     --remove-label run-r9700     >/dev/null 2>&1 || true

  if ! gh pr edit "${PR_NUMBER}"     --repo "${REPO}"     --add-label run-r9700     >/dev/null; then
    echo "cannot arm PR-label workflow; using local exact-SHA path" >&2
    return 1
  fi

  for _ in $(seq 1 12); do
    sleep 2

    local workflow_run
    workflow_run="$(
      find_matching_run "${EXACT_SHA}"
    )"

    if [[ -n "${workflow_run}" ]]; then
      echo "R9700 workflow accepted by GitHub:"
      printf '%s\n' "${workflow_run}"
      echo "The self-hosted service now owns execution."
      return 0
    fi
  done

  echo "GitHub did not instantiate the workflow; using local exact-SHA path" >&2
  return 1
}

if try_actions_path; then
  exit 0
fi

echo "== Local isolated exact-SHA execution =="

mkdir -p "${WORKTREE_ROOT}"
WORKTREE="${WORKTREE_ROOT}/${EXACT_SHA}"

git fetch origin "${EXACT_SHA}"   || git fetch origin "${PR_BRANCH}"

if [[ -e "${WORKTREE}/.git" ]]; then
  actual="$(
    git -C "${WORKTREE}"       rev-parse HEAD
  )"
  if [[ "${actual}" != "${EXACT_SHA}" ]]; then
    echo "existing worktree has wrong SHA: ${actual}" >&2
    exit 68
  fi
else
  rm -rf "${WORKTREE}"
  git worktree add     --detach     "${WORKTREE}"     "${EXACT_SHA}"
fi

if [[ -n "${SHADOW_EXPORT}" ]]; then
  PYTHONPATH="${WORKTREE}/src"   bash "${WORKTREE}/scripts/run-r9700-acceptance-slice.sh"     "${EXACT_SHA}"     "${SHADOW_EXPORT}"
else
  PYTHONPATH="${WORKTREE}/src"   bash "${WORKTREE}/scripts/run-r9700-acceptance-slice.sh"     "${EXACT_SHA}"
fi

echo
echo "Local exact-SHA acceptance finished in:"
echo "${WORKTREE}"
