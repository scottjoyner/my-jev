from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run-system-one-receipt-physical-acceptance.sh"
WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "system-one-receipt-physical-acceptance.yml"
)


def test_physical_receipt_acceptance_script_has_valid_bash_syntax():
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_physical_receipt_workflow_is_manual_only_and_r9700_scoped():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in text
    assert "pull_request:" not in text
    assert "\npush:" not in text
    assert "- self-hosted" in text
    assert "- linux" in text
    assert "runner_label" in text
    assert "cancel-in-progress: false" in text


def test_physical_receipt_acceptance_is_loopback_ephemeral_and_fail_closed():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--host 127.0.0.1" in text
    assert "MY_JEV_RECEIPT_ACCEPTANCE_PORT:-18088" in text
    assert "0.0.0.0" not in text
    assert 'trap cleanup EXIT' in text
    assert '"actions_allowed": false' in text
    assert '"local_writes_allowed": false' in text
    assert '"external_actions_allowed": false' in text
    assert '"privileged_actions_allowed": false' in text
    assert '"approval_gate_available": false' in text
    assert "authoritative_behavior_changed" in text


def test_physical_receipt_acceptance_pins_exact_consumer_and_sanitizes_model_id():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "AUTO_ASSIST_SHA" in text
    assert "system_one_receipt.py" in text
    assert "importlib.util.spec_from_file_location" in text
    assert 'provider.get("provider_id") != "my-jev"' in text
    assert 'provider.get("model_id") != expected_model_id' in text
    assert 'provider.get("model_artifact_sha256") != expected_artifact_sha' in text
    assert '"model_id_sha256": model_id_sha' in text
    assert '"model_id": expected_model_id' not in text


def test_physical_receipt_acceptance_requires_private_runner_local_checkpoint_config():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "system-one-receipt-acceptance.env" in text
    assert "mode & 0o022" in text
    assert "MY_JEV_RECEIPT_ACCEPTANCE_CHECKPOINT" in text
    assert "MY_JEV_RECEIPT_ACCEPTANCE_CALIBRATION" in text
    assert "model.pt" in text
