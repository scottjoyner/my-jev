"""Resolver tests for the advisory emitter.

Everything that touches a checkpoint, cache or ledger file is replaced here: the
regression under test is which number gets written into the evidence ledger, not
whether the model can be loaded.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from my_jev.trading.dataset import LabelConfig
from my_jev.trading.windows import WindowSet

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "emit_trading_advisory.py"


@pytest.fixture(scope="module")
def emitter():
    spec = importlib.util.spec_from_file_location("emit_trading_advisory", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeLedger:
    """In-memory stand-in that records what a real ledger would append."""

    def __init__(self, records: list[dict]) -> None:
        self.records = records
        self.resolved: list[tuple[str, dict]] = []

    def read(self) -> list[dict]:
        return list(self.records)

    def unresolved(self) -> list[str]:
        done = {aid for aid, _ in self.resolved}
        return [
            r["advisory_id"]
            for r in self.records
            if r["kind"] == "advisory" and r["advisory_id"] not in done
        ]

    def resolve(self, advisory_id: str, outcome: dict) -> None:
        self.resolved.append((advisory_id, outcome))


def _window_set(net: list[float], products: list[str], times: list[pd.Timestamp]) -> WindowSet:
    n = len(net)
    return WindowSet(
        features=np.zeros((n, 1)),
        labels=np.zeros(n, dtype=np.int64),
        net_forward=np.asarray(net, dtype=float),
        timestamps=np.asarray(times, dtype=object),
        product_ids=np.asarray(products, dtype=object),
        asset_start=np.arange(n),
        position_in_asset=np.zeros(n, dtype=np.int64),
        columns=("x",),
        mean=np.zeros(1),
        scale=np.ones(1),
    )


def _patch_io(monkeypatch, emitter, data: WindowSet, ledger: FakeLedger) -> None:
    monkeypatch.setattr(emitter, "AdvisoryLedger", lambda _path: ledger)
    monkeypatch.setattr(
        emitter,
        "load_model",
        lambda _run: (
            None,
            None,
            {"horizon_bars": 1, "round_trip_bps": 40.0},
            [],
            "digest",
        ),
    )
    monkeypatch.setattr(
        emitter,
        "build_dataset",
        lambda _root, _config, _columns: (
            data,
            LabelConfig(horizon_bars=1, round_trip_bps=40.0),
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["emit_trading_advisory.py", "--run", "dummy", "--resolve"],
    )


def test_resolve_scores_the_advised_asset_at_a_shared_timestamp(monkeypatch, emitter):
    """Two assets share a bar; the outcome must come from the advised one."""
    asof = pd.Timestamp("2025-01-02T00:00:00Z")
    data = _window_set(
        net=[0.012, -0.030],
        products=["BTC-USD", "DOGE-USD"],
        times=[asof, asof],
    )
    advisory = {
        "kind": "advisory",
        "advisory_id": f"BTC-USD@{asof}",
        "product_id": "BTC-USD",
        "asof": str(asof),
    }
    ledger = FakeLedger([advisory])
    _patch_io(monkeypatch, emitter, data, ledger)

    assert emitter.main() == 0
    assert ledger.resolved == [
        (
            advisory["advisory_id"],
            {
                "product_id": "BTC-USD",
                "asof": advisory["asof"],
                "forward_return": pytest.approx(0.016),  # 0.012 net + 40bp
                "net_forward_return": pytest.approx(0.012),
                "realised_at": str(asof + pd.Timedelta(hours=1)),
            },
        )
    ]


def test_resolve_leaves_missing_bar_unresolved(monkeypatch, emitter):
    """No matching (product, timestamp) bar means no fabricated outcome."""
    asof = pd.Timestamp("2025-01-02T00:00:00Z")
    data = _window_set(net=[0.012], products=["BTC-USD"], times=[asof])
    advisory = {
        "kind": "advisory",
        "advisory_id": f"SOL-USD@{asof}",
        "product_id": "SOL-USD",
        "asof": str(asof),
    }
    ledger = FakeLedger([advisory])
    _patch_io(monkeypatch, emitter, data, ledger)

    assert emitter.main() == 0
    assert ledger.resolved == []
    assert ledger.unresolved() == [advisory["advisory_id"]]
