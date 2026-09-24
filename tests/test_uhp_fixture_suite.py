import json

from my_jev.uhp_fixture_suite import main


def test_fixture_suite_renders_expected_consumer_cases(tmp_path):
    rc = main(
        [
            "--output-dir",
            str(tmp_path),
            "--now",
            "2026-09-23T22:30:00Z",
        ]
    )
    assert rc == 0

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["evidence_only"] is True
    assert manifest["runtime_authority_changed"] is False
    assert {
        name: case["expected"]
        for name, case in manifest["cases"].items()
    } == {
        "valid.json": "consumed",
        "expired.json": "expired",
        "authority-bearing.json": "authority_mutation_allowed",
        "model-fallback.json": "model_fallback",
        "handoff.json": "system_one_handoff",
    }

    valid = (tmp_path / "valid.json").read_text(encoding="utf-8")
    assert "r9700-primary" not in valid
    assert "x1-370" not in valid
    assert "eligible:opaque:r9700-a" in valid
    assert "eligible:opaque:cpu-b" in valid

    authority = json.loads(
        (tmp_path / "authority-bearing.json").read_text(encoding="utf-8")
    )
    assert (
        authority["metadata"]["hermes_system_one"]["authority"]["mutation_allowed"]
        is True
    )

    fallback = json.loads((tmp_path / "model-fallback.json").read_text(encoding="utf-8"))
    assert fallback["metadata"]["model_fallback"] is True

    handoff = json.loads((tmp_path / "handoff.json").read_text(encoding="utf-8"))
    assert handoff["status"] == "incomplete"
    assert handoff["incomplete_details"]["reason"] == "no_confident_action"
