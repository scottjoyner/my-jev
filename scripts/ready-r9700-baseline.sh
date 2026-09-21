#!/usr/bin/env bash
set -euo pipefail

REPO="${MY_JEV_GITHUB_REPO:-scottjoyner/my-jev}"
PR_NUMBER="${MY_JEV_PR_NUMBER:-1}"
RUNNER_LABEL="${MY_JEV_R9700_RUNNER_LABEL:-r9700}"
WORKTREE_ROOT="${MY_JEV_R9700_WORKTREE_ROOT:-$HOME/my-jev-r9700-worktrees}"
SHADOW_EXPORT="${1:-}"

ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd
)"
cd "${ROOT}"

for command in git gh python cut; do
  command -v "${command}" >/dev/null || {
    echo "${command} is required" >&2
    exit 64
  }
done

gh auth status >/dev/null 2>&1 || {
  echo "gh is not authenticated; run gh auth login first" >&2
  exit 65
}

find_matching_run() {
  local exact_sha="$1"

  gh run list     --repo "${REPO}"     --limit 50     --json databaseId,headSha,name,status,conclusion,url     | EXACT_SHA="${exact_sha}" python -c '
import json
import os
import sys

exact_sha = os.environ["EXACT_SHA"]
runs = json.load(sys.stdin)

for run in runs:
    if (
        run.get("name") == "r9700-baseline"
        and run.get("headSha") == exact_sha
    ):
        fields = [
            str(run.get("databaseId") or ""),
            str(run.get("status") or ""),
            str(run.get("conclusion") or ""),
            str(run.get("url") or ""),
        ]
        print("\t".join(fields))
        break
'
}

echo "== R9700 host/runner preflight =="
MY_JEV_GITHUB_REPO="${REPO}" MY_JEV_R9700_RUNNER_LABEL="${RUNNER_LABEL}" bash scripts/bootstrap-r9700-github-runner.sh

if [[ -n "${SHADOW_EXPORT}" && ! -f "${SHADOW_EXPORT}" ]]; then
  echo "shadow export does not exist: ${SHADOW_EXPORT}" >&2
  exit 67
fi

if [[ "${RUNNER_LABEL}" != "r9700" ]]; then
  gh variable set MY_JEV_R9700_RUNNER_LABEL     --repo "${REPO}"     --body "${RUNNER_LABEL}"
fi

if [[ -n "${SHADOW_EXPORT}" ]]; then
  gh variable set MY_JEV_ASSISTX_SHADOW_EXPORT     --repo "${REPO}"     --body "${SHADOW_EXPORT}"
fi

EXACT_SHA="$(
  gh pr view "${PR_NUMBER}"     --repo "${REPO}"     --json headRefOid     --jq .headRefOid
)"

if [[ ! "${EXACT_SHA}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "could not resolve exact PR head SHA" >&2
  exit 66
fi

echo "repository: ${REPO}"
echo "pr: #${PR_NUMBER}"
echo "exact_sha: ${EXACT_SHA}"
echo "runner_label: ${RUNNER_LABEL}"

existing_run="$(find_matching_run "${EXACT_SHA}")"

if [[ -n "${existing_run}" ]]; then
  existing_status="$(
    printf '%s\n' "${existing_run}" | cut -f2
  )"
  existing_conclusion="$(
    printf '%s\n' "${existing_run}" | cut -f3
  )"

  if [[ "${existing_status}" != "completed" || "${existing_conclusion}" == "success" ]]; then
    echo "matching R9700 workflow already exists:"
    printf '%s\n' "${existing_run}"
    exit 0
  fi

  echo "previous exact-SHA R9700 run did not succeed:"
  printf '%s\n' "${existing_run}"
  echo "re-arming the guarded workflow"
fi

echo "== Arming guarded PR-label workflow =="

gh label create run-r9700   --repo "${REPO}"   --description "Explicitly run exact-SHA R9700 baseline acceptance"   --color B60205   --force

current_sha="$(
  gh pr view "${PR_NUMBER}"     --repo "${REPO}"     --json headRefOid     --jq .headRefOid
)"
if [[ "${current_sha}" != "${EXACT_SHA}" ]]; then
  echo "PR head moved during readiness setup; restarting with the new head"
  exec bash "$0" "${SHADOW_EXPORT}"
fi

gh pr edit "${PR_NUMBER}"   --repo "${REPO}"   --remove-label run-r9700   >/dev/null 2>&1 || true

gh pr edit "${PR_NUMBER}"   --repo "${REPO}"   --add-label run-r9700   >/dev/null

for _ in $(seq 1 12); do
  sleep 2

  workflow_run="$(find_matching_run "${EXACT_SHA}")"
  if [[ -n "${workflow_run}" ]]; then
    echo "R9700 workflow accepted by GitHub:"
    printf '%s\n' "${workflow_run}"
    echo
    echo "The self-hosted service now owns execution; this shell can be closed."
    exit 0
  fi
done

echo "GitHub did not instantiate the PR-label workflow."
echo "Falling back to isolated local exact-SHA execution."

mkdir -p "${WORKTREE_ROOT}"
WORKTREE="${WORKTREE_ROOT}/${EXACT_SHA}"

git fetch origin "${EXACT_SHA}"

if [[ -e "${WORKTREE}/.git" ]]; then
  actual="$(
    git -C "${WORKTREE}" rev-parse HEAD
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
