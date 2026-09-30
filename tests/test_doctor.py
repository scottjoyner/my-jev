from my_jev.doctor import (
    hardware_report,
    validate_environment,
)


def test_doctor_report_has_runtime_fields():
    report = hardware_report()

    assert "torch_version" in report
    assert "hip_version" in report
    assert "accelerator_available" in report
    assert "bf16_supported" in report
    assert "packages" in report


def test_doctor_validation_reports_missing_requirements():
    report = {
        "accelerator_available": False,
        "bf16_supported": False,
        "packages": {
            "transformers": "5.5.0",
            "peft": None,
            "accelerate": None,
        },
    }
    issues = validate_environment(
        report,
        require_gpu=True,
        require_bf16=True,
        require_causal=True,
    )

    assert any(
        "accelerator" in issue
        for issue in issues
    )
    assert any(
        "BF16" in issue
        for issue in issues
    )
    assert any(
        "peft" in issue
        for issue in issues
    )
    assert any(
        "accelerate" in issue
        for issue in issues
    )
