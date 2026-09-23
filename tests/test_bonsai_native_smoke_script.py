from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "run-bonsai-native-smoke.sh"
)


def test_bonsai_native_smoke_script_is_valid_bash():
    subprocess.run(
        [
            "bash",
            "-n",
            str(
                SCRIPT
            ),
        ],
        cwd=ROOT,
        check=True,
    )


def test_bonsai_native_smoke_orders_probe_build_capture_and_head():
    text = SCRIPT.read_text(
        encoding="utf-8"
    )

    probe = text.index(
        "my_jev.bonsai_probe"
    )
    build = text.index(
        "build-bonsai-bridge.sh"
    )
    capture = text.index(
        "my_jev.bonsai_native"
    )
    head = text.index(
        "my_jev.bonsai_smoke"
    )

    assert (
        probe
        < build
        < capture
        < head
    )


def test_bonsai_native_smoke_can_generate_deterministic_record():
    text = SCRIPT.read_text(
        encoding="utf-8"
    )

    assert (
        "generate_agent_policy_records"
        in text
    )
    assert (
        "seed=137"
        in text
    )
    assert (
        "bonsai-native-smoke-receipt.json"
        in text
    )


def test_bonsai_native_smoke_preserves_authority_boundary():
    text = SCRIPT.read_text(
        encoding="utf-8"
    )

    assert (
        '"dispatch_allowed": False'
        in text
    )
    assert (
        '"runtime_authority_changed": False'
        in text
    )
    assert (
        '"quality_validated": False'
        in text
    )
