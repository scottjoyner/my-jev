import time

from my_jev.registry import (
    ExperimentEntry,
    ExperimentRegistry,
)


def _entry(
    run_id,
    *,
    created_at,
    promoted=False,
    parent=None,
):
    return ExperimentEntry(
        run_id=run_id,
        experiment="assistx-policy",
        created_at=created_at,
        updated_at=created_at,
        status=(
            "candidate"
            if promoted
            else "rejected"
        ),
        spec_path="spec.toml",
        spec_sha256="a" * 64,
        model_backend="encoder_option_query",
        backbone="test-backbone",
        train_sha256="1" * 64,
        validation_sha256="2" * 64,
        calibration_sha256="3" * 64,
        test_sha256="4" * 64,
        run_dir=f"runs/{run_id}",
        promoted=promoted,
        parent_run_id=parent,
    )


def test_registry_tracks_latest_promoted_and_lineage(tmp_path):
    registry = ExperimentRegistry(
        tmp_path / "registry.json"
    )
    first = registry.register(
        _entry(
            "run-1",
            created_at=time.time(),
            promoted=True,
        )
    )
    second = registry.register(
        _entry(
            "run-2",
            created_at=first.created_at + 1,
            promoted=False,
            parent="run-1",
        )
    )

    assert (
        registry.latest(
            "assistx-policy"
        ).run_id
        == second.run_id
    )
    assert (
        registry.latest(
            "assistx-policy",
            promoted_only=True,
        ).run_id
        == first.run_id
    )
    assert [
        item.run_id
        for item in registry.lineage(
            "run-2"
        )
    ] == ["run-2", "run-1"]


def test_registry_persists_updates(tmp_path):
    path = tmp_path / "registry.json"
    registry = ExperimentRegistry(path)
    registry.register(
        _entry(
            "run-1",
            created_at=1.0,
        )
    )
    registry.update(
        "run-1",
        status="failed",
        notes=["boom"],
    )

    reloaded = ExperimentRegistry(path)
    assert (
        reloaded.get(
            "run-1"
        ).status
        == "failed"
    )
    assert reloaded.get(
        "run-1"
    ).notes == ["boom"]
