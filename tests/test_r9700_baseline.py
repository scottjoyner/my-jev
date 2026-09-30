from __future__ import annotations

import subprocess
from pathlib import Path

from my_jev.experiment import (
    _train_command,
    load_experiment_spec,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "assistx-modernbert-r9700-baseline.toml"
)
LAUNCHER = (
    ROOT
    / "scripts"
    / "run-r9700-modernbert-baseline.sh"
)


def _value_after(command: list[str], flag: str) -> str:
    return command[
        command.index(flag)
        + 1
    ]


def test_r9700_baseline_spec_is_small_full_encoder_run():
    spec, spec_sha = load_experiment_spec(
        SPEC_PATH
    )

    assert len(spec_sha) == 64
    assert spec.experiment.name == (
        "assistx-policy-modernbert-r9700-baseline"
    )
    assert spec.experiment.parent == ""

    assert spec.model.backend == (
        "encoder_option_query"
    )
    assert spec.model.backbone == (
        "answerdotai/ModernBERT-base"
    )
    assert spec.model.head_kind == (
        "option_query"
    )
    assert spec.model.head_rank == 256

    assert spec.train.epochs == 1
    assert spec.train.batch_size == 2
    assert spec.train.grad_accum == 8
    assert spec.train.max_state_length == 2048
    assert spec.train.max_candidate_length == 192
    assert spec.train.bf16 is True
    assert spec.train.freeze_backbone is False
    assert (
        spec.train.gradient_checkpointing
        is True
    )

    assert spec.data.train.startswith(
        "runs/datasets/"
    )
    assert spec.data.validation.startswith(
        "runs/datasets/"
    )
    assert spec.data.calibration.startswith(
        "runs/datasets/"
    )
    assert spec.data.test.startswith(
        "runs/datasets/"
    )

    command = _train_command(
        spec,
        ROOT,
        ROOT / "runs" / "test-r9700",
    )
    assert "my_jev.train" in command
    assert "--freeze-backbone" not in command
    assert "--bf16" in command
    assert "--gradient-checkpointing" in command
    assert _value_after(
        command,
        "--epochs",
    ) == "1"
    assert _value_after(
        command,
        "--batch-size",
    ) == "2"
    assert _value_after(
        command,
        "--grad-accum",
    ) == "8"
    assert _value_after(
        command,
        "--max-state-length",
    ) == "2048"
    assert _value_after(
        command,
        "--max-candidate-length",
    ) == "192"


def test_r9700_launcher_is_valid_bash_and_requires_sha():
    subprocess.run(
        ["bash", "-n", str(LAUNCHER)],
        cwd=ROOT,
        check=True,
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 64
    assert "40-char git SHA" in result.stderr


def test_r9700_launcher_rejects_wrong_exact_sha_before_gpu_work():
    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "0" * 40,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 65
    assert "HEAD" in result.stderr
    assert "expected" in result.stderr


def test_r9700_launcher_keeps_authority_out_of_scope():
    text = LAUNCHER.read_text(
        encoding="utf-8"
    )

    assert "my_jev.agent_policy" not in text
    assert "my_jev.assistx_adapter" not in text
    assert "external_actions_allowed" not in text
    assert "privileged_actions_allowed" not in text
    assert "approval_gate_available" not in text

    assert "my_jev.doctor" in text
    assert "my_jev.prepare" in text
    assert "my_jev.experiment" in text
    assert "r9700-baseline-receipt.json" in text
