from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = (
    ROOT
    / "scripts"
    / "bootstrap-r9700-github-runner.sh"
)
READY = (
    ROOT
    / "scripts"
    / "ready-r9700-baseline.sh"
)


def test_runner_bootstrap_is_valid_bash():
    subprocess.run(
        [
            "bash",
            "-n",
            str(BOOTSTRAP),
        ],
        cwd=ROOT,
        check=True,
    )


def test_ready_launcher_is_valid_bash():
    subprocess.run(
        [
            "bash",
            "-n",
            str(READY),
        ],
        cwd=ROOT,
        check=True,
    )


def test_runner_bootstrap_checks_rocm_and_r9700_before_registration():
    text = BOOTSTRAP.read_text(
        encoding="utf-8"
    )

    assert "torch.version.hip" in text
    assert "torch.cuda.is_available()" in text
    assert '"r9700" in name.lower()' in text
    assert "MY_JEV_R9700_MIN_GIB" in text
    assert (
        "actions/runners/registration-token"
        in text
    )


def test_runner_bootstrap_uses_required_workflow_label_and_service():
    text = BOOTSTRAP.read_text(
        encoding="utf-8"
    )

    assert (
        'MY_JEV_R9700_RUNNER_LABEL:-r9700'
        in text
    )
    assert (
        '--labels "${RUNNER_LABEL}"'
        in text
    )
    assert "sudo ./svc.sh install" in text
    assert "sudo ./svc.sh start" in text
    assert (
        "runner is online with the required "
        "label and currently busy"
        in text
    )


def test_runner_bootstrap_does_not_print_registration_token():
    text = BOOTSTRAP.read_text(
        encoding="utf-8"
    )

    assert "REGISTRATION_TOKEN" in text
    assert "unset REGISTRATION_TOKEN" in text
    assert (
        "echo ${REGISTRATION_TOKEN}"
        not in text
    )
    assert (
        "printf '%s' ${REGISTRATION_TOKEN}"
        not in text
    )


def test_ready_launcher_resolves_exact_pr_head_and_arms_guarded_workflow():
    text = READY.read_text(
        encoding="utf-8"
    )

    assert "gh pr view" in text
    assert "--json headRefOid" in text
    assert "bootstrap-r9700-github-runner.sh" in text
    assert "gh label create" in text
    assert "run-r9700" in text
    assert "gh pr edit" in text
    assert "r9700-baseline" in text


def test_ready_launcher_preserves_current_checkout_with_detached_worktree_fallback():
    text = READY.read_text(
        encoding="utf-8"
    )

    assert "git worktree add" in text
    assert "--detach" in text
    assert (
        "my-jev-r9700-worktrees"
        in text
    )
    assert (
        "run-r9700-acceptance-slice.sh"
        in text
    )
    assert "git checkout" not in text
    assert "git switch" not in text


def test_ready_launcher_reuses_live_or_successful_run_but_rearms_failure():
    text = READY.read_text(
        encoding="utf-8"
    )

    assert (
        'existing_status}" != "completed"'
        in text
    )
    assert (
        'existing_conclusion}" == "success"'
        in text
    )
    assert (
        "re-arming the guarded workflow"
        in text
    )
