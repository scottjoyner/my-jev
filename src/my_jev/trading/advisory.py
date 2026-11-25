"""Non-dispatch advisory emission for the trading decision head.

Why a NO-GO still warrants an emitter
-------------------------------------
Validation on 2026-10-05 returned NO-GO: across four walk-forward folds with
non-overlapping positions, aggregate skill was -1.0bp with a 95% CI of
[-59.5, +63.0] and ``P(skill > 0) = 0.48``. That is not evidence of an edge, and
it is also not proof of absence — it is **absence of evidence**, on a dataset
that yields only ~418 independent bets.

So the useful next step is not to trade it and not to bin it. It is to keep
emitting scored opinions into a ledger so that forward evidence accumulates, and
to revisit the verdict when there are enough of them. An emitter that refuses to
be wired to capital is exactly the right shape for that job.

Authority boundary
------------------
This module emits an **opinion**, never an instruction. Every advisory carries
explicit false assertions for the fields a downstream system would need in order
to act:

* ``dispatch_allowed = False``
* ``approval_granted = False``

and exposes no order, size, or venue field at all. The probability distribution
is over ``buy`` / ``sell`` / ``hold``; turning it into a position is somebody
else's decision, made outside this package, under whatever controls that system
already has. This mirrors the advisory boundary already established for the
System-One lane in ``docs/BONSAI2_DECISION_HEAD.md``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


#: Schema version for the advisory ledger. Bump if field semantics change.
ADVISORY_SCHEMA_VERSION = 1

#: The boundary is asserted, not documented. Anything consuming this ledger
#: should treat a missing/false pair as a hard stop rather than a warning.
AUTHORITY_ASSERTIONS: dict[str, bool] = {
    "dispatch_allowed": False,
    "approval_granted": False,
}


@dataclass(frozen=True)
class TradingAdvisory:
    """One scored, non-dispatchable opinion about a market state."""

    product_id: str
    asof: str
    schema_version: int
    options: tuple[str, ...]
    probabilities: dict[str, float]
    top_option: str
    top_probability: float
    # What the model is *not* allowed to do with this. Redundant on purpose: a
    # consumer that deserialises into a loose dict will still see them.
    dispatch_allowed: bool = False
    approval_granted: bool = False
    # Provenance, so a later review can tell which weights produced the opinion.
    model_sha256: str = ""
    feature_schema: tuple[str, ...] = ()
    horizon_bars: int = 0
    round_trip_bps: float = 0.0
    #: Set when the model has not been validated. "unvalidated" must not be
    #: silently treated as "validated".
    validation_state: str = "unvalidated"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["options"] = list(self.options)
        payload["feature_schema"] = list(self.feature_schema)
        payload["emitted_unix"] = time.time()
        return json.dumps(payload, sort_keys=True, default=str)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["options"] = list(self.options)
        payload["feature_schema"] = list(self.feature_schema)
        payload["emitted_unix"] = time.time()
        return payload


class AdvisoryLedger:
    """Append-only JSONL ledger of advisories and their eventual outcomes.

    Outcomes are appended separately and joined on ``advisory_id``. Keeping
    prediction and resolution in one file per record would make it easy to score
    only the opinions that looked good; a separate append-only outcome stream
    means every emitted opinion must be resolved, and a missing one is visible.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, advisory: TradingAdvisory, advisory_id: str) -> str:
        if advisory.dispatch_allowed or advisory.approval_granted:
            raise ValueError(
                "an advisory may never assert dispatch or approval authority; "
                "that is the whole point of this lane"
            )
        record = advisory.as_dict()
        record["advisory_id"] = advisory_id
        record["kind"] = "advisory"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        return advisory_id

    def resolve(self, advisory_id: str, outcome: dict[str, Any]) -> None:
        """Append the realised outcome for a previously emitted advisory."""
        record = {
            "kind": "outcome",
            "advisory_id": advisory_id,
            "resolved_unix": time.time(),
            **outcome,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def unresolved(self) -> list[str]:
        """Advisory ids with no outcome — the open evidence set."""
        emitted: list[str] = []
        resolved: set[str] = set()
        for record in self.read():
            if record.get("kind") == "advisory":
                emitted.append(record["advisory_id"])
            elif record.get("kind") == "outcome":
                resolved.add(record["advisory_id"])
        return [a for a in emitted if a not in resolved]

    def score(self, *, cost_bps: float) -> dict[str, float]:
        """Score every *resolved* advisory. Unresolved ones are excluded, not dropped.

        Reported as coverage alongside the metrics so a high score computed from
        a handful of resolved opinions cannot be mistaken for a large sample.
        """
        advisories = {
            r["advisory_id"]: r for r in self.read() if r.get("kind") == "advisory"
        }
        rows = []
        for record in self.read():
            if record.get("kind") != "outcome":
                continue
            origin = advisories.get(record["advisory_id"])
            if origin is None:
                continue
            forward = float(record.get("forward_return", 0.0))
            probs = origin.get("probabilities", {})
            signed = float(probs.get("buy", 0.0)) - float(probs.get("sell", 0.0))
            size = abs(signed)
            if size < 0.10:
                continue
            pnl = signed * forward - size * (cost_bps / 10_000.0)
            rows.append(
                {
                    "signed": signed,
                    "forward": forward,
                    "pnl": pnl,
                    "hit": float(np.sign(signed) == np.sign(forward)),
                }
            )
        total_emitted = len(advisories)
        if not rows:
            return {
                "emitted": total_emitted,
                "scored": 0,
                "coverage": 0.0,
                "hit_rate": float("nan"),
                "mean_net_bp": float("nan"),
            }
        pnl = np.array([r["pnl"] for r in rows])
        return {
            "emitted": total_emitted,
            "scored": len(rows),
            "coverage": len(rows) / total_emitted if total_emitted else 0.0,
            "hit_rate": float(np.mean([r["hit"] for r in rows])),
            "mean_net_bp": float(pnl.mean() * 1e4),
            "sharpe": float(pnl.mean() / (pnl.std(ddof=0) or 1e-12)),
        }


def advisory_id(product_id: str, asof: str) -> str:
    """Deterministic id so re-running an emitter does not double-count a bar."""
    return f"{product_id}@{asof}"
