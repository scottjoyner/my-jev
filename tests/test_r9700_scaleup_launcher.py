from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "scripts"
    / "run-r9700-scaleup-if-ready.sh"
)


def test_scaleup_launcher_is_valid_bash_and_requires_args():
    subprocess.run(
        [
            "bash",
            "-n",
            str(LAUNCHER),
        ],
        cwd=ROOT,
        check=True,
    )

    result = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 64


def test_scaleup_launcher_requires_passing_readiness_before_gpu_run():
    text = LAUNCHER.read_text(
        encoding="utf-8"
    )

    assert (
        "r9700-scale-readiness.json"
        in text
    )
    assert (
        "scale-up readiness is HOLD"
        in text
    )
    assert (
        'payload.get("decision") != "scale"'
        in text
    )
    assert (
        "my_jev.doctor"
        in text
    )
    assert (
        "my_jev.experiment"
        in text
    )


def test_scaleup_launcher_pins_50k_dataset_contract():
    text = LAUNCHER.read_text(
        encoding="utf-8"
    )

    assert "--records 50000" in text
    assert "--seed 23" in text
    assert (
        "assistx-policy-r9700-scaleup-v1"
        in text
    )
    assert (
        "GENERATOR_VERSION"
        in text
    )


def test_scaleup_launcher_preserves_authority_boundary_and_parent_receipt():
    text = LAUNCHER.read_text(
        encoding="utf-8"
    )

    assert (
        "r9700-scaleup-receipt.json"
        in text
    )
    assert (
        '"evidence_only": True'
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
    assert (
        '"parent_scale_readiness"'
        in text
    )
