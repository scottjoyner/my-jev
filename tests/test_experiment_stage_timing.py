from __future__ import annotations

import json
import sys

import pytest

from my_jev.experiment import _stage


def _read(run_dir, name):
    return json.loads((run_dir / "stages" / f"{name}.timing.json").read_text())


def test_a_stage_records_its_duration(tmp_path):
    """The source of every real per-stage measurement the planner can use."""
    _stage(
        "train",
        [sys.executable, "-c", "import time; time.sleep(0.2)"],
        run_dir=tmp_path,
        dry_run=False,
    )
    timing = _read(tmp_path, "train")
    assert timing["name"] == "train"
    assert timing["completed"] is True
    assert timing["duration_seconds"] > 0.0
    assert timing["finished_at"] >= timing["started_at"]


def test_a_failing_stage_still_records_its_duration(tmp_path):
    """A stage that died consumed GPU. Losing that number is how estimates rot."""
    with pytest.raises(RuntimeError):
        _stage(
            "benchmark",
            [sys.executable, "-c", "raise SystemExit(3)"],
            run_dir=tmp_path,
            dry_run=False,
        )
    timing = _read(tmp_path, "benchmark")
    assert timing["completed"] is False
    assert timing["duration_seconds"] >= 0.0


def test_a_dry_run_records_no_timing(tmp_path):
    """It never started, so a duration for it would be pure fiction."""
    _stage(
        "train",
        [sys.executable, "-c", "raise SystemExit(1)"],
        run_dir=tmp_path,
        dry_run=True,
    )
    assert not (tmp_path / "stages" / "train.timing.json").exists()
    assert (tmp_path / "stages" / "train.command.json").exists()


def test_the_planner_reads_what_the_instrumentation_writes(tmp_path):
    """The two halves have to agree, or the measurement is decoration."""
    from my_jev.gpu_plan import measured_stage_minutes_from_run

    _stage(
        "train",
        [sys.executable, "-c", "import time; time.sleep(0.2)"],
        run_dir=tmp_path,
        dry_run=False,
    )
    measured = measured_stage_minutes_from_run(tmp_path)
    from my_jev.gpu_plan import StageKind

    assert StageKind.train in measured
    assert measured[StageKind.train] > 0.0
    # And a failed stage is excluded from costing.
    with pytest.raises(RuntimeError):
        _stage(
            "benchmark",
            [sys.executable, "-c", "raise SystemExit(1)"],
            run_dir=tmp_path,
            dry_run=False,
        )
    assert StageKind.benchmark not in measured_stage_minutes_from_run(tmp_path)
