from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (
    ROOT
    / "scripts"
    / "run-r9700-acceptance-slice.sh"
)
WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "r9700-baseline.yml"
)


def test_acceptance_wrapper_is_valid_bash_and_requires_sha():
    subprocess.run(
        [
            "bash",
            "-n",
            str(WRAPPER),
        ],
        cwd=ROOT,
        check=True,
    )

    result = subprocess.run(
        [
            "bash",
            str(WRAPPER),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 64
    assert "40-char git SHA" in result.stderr


def test_acceptance_wrapper_binds_baseline_validation_and_shadow():
    text = WRAPPER.read_text(
        encoding="utf-8"
    )

    assert (
        "run-r9700-modernbert-baseline.sh"
        in text
    )
    assert (
        "my_jev.baseline_evidence"
        in text
    )
    assert (
        "my_jev.shadow_evaluation"
        in text
    )
    assert (
        "r9700-acceptance-summary.json"
        in text
    )
    assert (
        '"runtime_authority_changed": False'
        in text
    )
    assert (
        '"dispatch_allowed": False'
        in text
    )


def test_r9700_workflow_is_manual_or_explicit_label_only():
    text = WORKFLOW.read_text(
        encoding="utf-8"
    )

    assert "workflow_dispatch:" in text
    assert "types:" in text
    assert "- labeled" in text
    assert (
        "github.event.label.name == "
        "'run-r9700'"
        in text
    )
    assert (
        "github.event.pull_request.head.repo.full_name "
        "== github.repository"
        in text
    )

    assert "push:" not in text
    assert "schedule:" not in text


def test_r9700_workflow_targets_explicit_self_hosted_gpu_label():
    text = WORKFLOW.read_text(
        encoding="utf-8"
    )

    assert "- self-hosted" in text
    assert "- linux" in text
    assert (
        "MY_JEV_R9700_RUNNER_LABEL"
        in text
    )
    assert (
        "default: r9700"
        in text
    )
    assert (
        "cancel-in-progress: false"
        in text
    )


def test_r9700_workflow_preserves_host_rocm_environment():
    text = WORKFLOW.read_text(
        encoding="utf-8"
    )

    assert (
        'PYTHONPATH: "${{ github.workspace }}/src"'
        in text
    )
    assert (
        "MY_JEV_R9700_PYTHON"
        in text
    )
    assert (
        'echo "PYTHON_BIN=${PYTHON_BIN}"'
        in text
    )
    assert (
        "torch.version.hip"
        in text
    )
    assert (
        "python -m pip install"
        not in text
    )
    assert (
        "actions/setup-python"
        not in text
    )
    assert (
        "my_jev import is not bound "
        "to exact checkout"
        in text
    )


def test_r9700_workflow_checks_out_exact_sha_and_uses_acceptance_wrapper():
    text = WORKFLOW.read_text(
        encoding="utf-8"
    )

    assert (
        'ref: "${{ env.EXACT_SHA }}"'
        in text
    )
    assert (
        'test "$(git rev-parse HEAD)" = '
        '"${EXACT_SHA}"'
        in text
    )
    assert (
        "run-r9700-acceptance-slice.sh"
        in text
    )
    assert (
        "actions/upload-artifact@v4"
        in text
    )
    assert (
        "r9700-evidence-${{ env.EXACT_SHA }}"
        in text
    )


def test_r9700_workflow_does_not_gain_runtime_authority():
    text = WORKFLOW.read_text(
        encoding="utf-8"
    )

    forbidden = (
        "external_actions_allowed",
        "privileged_actions_allowed",
        "approval_gate_available",
        "my_jev.assistx_adapter",
        "my_jev.agent_policy",
    )
    assert not any(
        item in text
        for item in forbidden
    )

    assert (
        "Runtime authority changed: `false`"
        in text
    )
    assert (
        "Dispatch allowed by this workflow: `false`"
        in text
    )
