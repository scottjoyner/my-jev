"""Shared test policy.

## A contended GPU is an environmental skip, not a failure

`train.main` and `train_causal.main` select `cuda` whenever a device is visible,
which is the right default for a training run and the wrong one for a test suite on a
shared machine. With three `llama-server` processes holding 30.5 of 31.9 GiB, one test
here died with `torch.OutOfMemoryError` -- a fact about the machine, reported as a
fact about the code.

Two things address that, in order of preference:

1. **Steer the test.** `--device cpu` now exists on both trainers, so a test that
   does not need the card never reaches for it. That is the real fix and it is what
   `test_the_training_loop_stops_on_a_non_finite_gradient` now does.
2. **Skip what still cannot run.** This hook is the safety net for anything that
   legitimately needs the card. It matches `torch.cuda.OutOfMemoryError` *only* --
   not `MemoryError`, and not `RuntimeError` generally -- because a generic handler
   here would convert a genuine defect in allocation logic into a green skip.

A skipped test is reported, so "the suite was green" still distinguishes "everything
ran" from "the card was busy".
"""

from __future__ import annotations

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when != "call" or not report.failed:
        return

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a base dependency
        return

    if not isinstance(call.excinfo.value, torch.cuda.OutOfMemoryError):
        return

    # Reported as a skip rather than raised. Calling `pytest.skip()` from inside this
    # hook raises during report generation and takes the whole session down with an
    # INTERNALERROR, which is worse than the failure it was replacing. Setting the
    # outcome and the reason is the supported way to do it.
    report.outcome = "skipped"
    # The terminal reporter unpacks a skipped report's longrepr as
    # (path, lineno, reason); handing it a bare string raises inside pytest's own
    # display code, which is a far worse outcome than the failure we are replacing.
    report.longrepr = (
        str(item.fspath),
        item.location[1],
        "skipped: the GPU is present but has no free memory "
        f"-- {call.excinfo.value}",
    )
