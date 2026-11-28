#!/usr/bin/env python3
"""Score market state with the decision head and append a non-dispatch advisory.

Deliberately a *ledger*, not a signal bus. Validation returned NO-GO (aggregate
skill -1.0bp, 95% CI [-59.5, +63.0], 418 independent bets), so the model's job
here is to accumulate forward evidence until there is enough to decide, not to
tell the trading system what to do.

What this writes
----------------
One JSONL line per scored bar: the option distribution, provenance (weights
hash, feature schema, horizon, cost), the validation state, and
``dispatch_allowed=false`` / ``approval_granted=false``. Nothing else. There is
no field a caller could mistake for an order.

Re-running is safe: advisory ids are ``product@asof``, so a repeated bar is
detectable rather than silently double-counted, and ``--resolve`` fills in
outcomes for previously emitted opinions once their horizon has elapsed.

Usage:
    # score the most recent complete bar for every cached asset
    python scripts/emit_trading_advisory.py --run runs/sweep/h168

    # later, resolve opinions whose horizon has elapsed
    python scripts/emit_trading_advisory.py --run runs/sweep/h168 --resolve

    # inspect the open evidence set
    python scripts/emit_trading_advisory.py --status --ledger data/advisories.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from my_jev.decision_attention import ExternalDecisionAttentionHead  # noqa: E402
from my_jev.trading import OPTIONS  # noqa: E402
from my_jev.trading.advisory import (  # noqa: E402
    AdvisoryLedger,
    TradingAdvisory,
    advisory_id,
)
from my_jev.trading.dataset import LabelConfig, load_universe  # noqa: E402
from my_jev.trading.encoder import MarketTapEncoder  # noqa: E402
from my_jev.trading.splits import calendar_boundaries, mask_windows  # noqa: E402
from my_jev.trading.windows import fit_scaler, make_batch, stack_series  # noqa: E402


def model_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(run: Path):
    blob = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    config = blob["config"]
    encoder = MarketTapEncoder(
        feature_count=len(blob["columns"]),
        hidden_size=int(config["hidden_size"]),
        depth=int(config["depth"]),
    ).eval()
    encoder.load_state_dict(blob["encoder"])
    head = ExternalDecisionAttentionHead(
        hidden_size=int(config["hidden_size"]), tap_count=4, rank=64
    ).eval()
    head.load_state_dict(blob["head"])
    return encoder, head, config, list(blob["columns"]), model_sha256(run / "model.pt")


def build_dataset(root: Path, config: dict, columns: list[str]):
    """Rebuild the exact features/scaling the checkpoint was trained on."""
    from train_trading_head import _subsample  # noqa: PLC0415

    label_config = LabelConfig(
        horizon_bars=int(config["horizon_bars"]),
        round_trip_bps=float(config["round_trip_bps"]),
    )
    series = _subsample(load_universe(root, config=label_config), 12000)
    data = stack_series(series, sequence_length=int(config["sequence_length"]))
    if list(data.columns) != list(columns):
        raise SystemExit(
            "feature schema drift: checkpoint was trained on a different feature "
            f"set ({len(columns)} cols vs {len(data.columns)} now). Refusing to "
            "score with mismatched inputs."
        )
    return data, label_config


def resolve_outcome(data, record: dict, *, cost: float, horizon_bars: int) -> dict | None:
    """Resolve one advisory against the forward return already stored per bar.

    ``WindowSet`` has no close column and is not a DataFrame; the bars it does
    carry are ``timestamps``, ``product_ids`` and ``net_forward`` (raw forward
    return minus the round-trip cost). Reusing that value keeps resolution on
    exactly the target the head was trained on, without re-reading price files.

    Matching is on *both* product and timestamp: at a shared timestamp the
    universe has one row per asset, so a time-only match can silently score a
    BTC opinion against a DOGE move. A missing or ambiguous bar is left
    unresolved rather than guessed at.
    """
    product_id = record["product_id"]
    asof = pd.Timestamp(record["asof"])
    rows = np.flatnonzero(
        (data.product_ids == product_id) & (data.timestamps == asof)
    )
    if len(rows) != 1:
        return None
    net = float(data.net_forward[int(rows[0])])
    if not np.isfinite(net):
        return None
    return {
        "product_id": product_id,
        "asof": record["asof"],
        # Undo the netting exactly as the training/evaluation lane does.
        "forward_return": net + cost,
        "net_forward_return": net,
        "realised_at": str(asof + pd.Timedelta(hours=horizon_bars)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="runs/<name> directory with model.pt")
    parser.add_argument("--root", type=Path, default=Path("data/trading"))
    parser.add_argument("--ledger", type=Path, default=Path("data/advisories.jsonl"))
    parser.add_argument("--validation-state", default="unvalidated")
    parser.add_argument("--resolve", action="store_true", help="resolve matured opinions")
    parser.add_argument("--status", action="store_true", help="print ledger status")
    args = parser.parse_args()

    ledger = AdvisoryLedger(args.ledger)

    if args.status:
        records = ledger.read()
        stats = ledger.score(cost_bps=40.0)
        print(f"ledger      : {args.ledger} ({len(records)} records)")
        print(f"emitted     : {stats['emitted']}")
        print(f"scored      : {stats['scored']}  (coverage {stats['coverage'] * 100:.1f}%)")
        print(f"open/unmet  : {len(ledger.unresolved())}")
        if stats["scored"]:
            print(
                f"hit rate    : {stats['hit_rate']:.3f}   "
                f"mean net    : {stats['mean_net_bp']:+.2f}bp   "
                f"sharpe      : {stats.get('sharpe', float('nan')):+.3f}"
            )
        else:
            print("no resolved outcomes yet")
        return 0

    if not args.run:
        parser.error("--run is required unless --status is given")

    encoder, head, config, columns, digest = load_model(args.run)
    data, label_config = build_dataset(args.root, config, columns)

    if args.resolve:
        unresolved = set(ledger.unresolved())
        resolved = 0
        for record in ledger.read():
            if record.get("kind") != "advisory":
                continue
            aid = record["advisory_id"]
            if aid not in unresolved:
                continue
            outcome = resolve_outcome(
                data,
                record,
                cost=label_config.cost,
                horizon_bars=int(config["horizon_bars"]),
            )
            if outcome is None:
                continue
            ledger.resolve(aid, outcome)
            resolved += 1
        print(f"resolved {resolved} matured opinion(s)")
        return 0

    # Emit for the most recent fully-warmed bar of each asset.
    boundaries = calendar_boundaries(data.timestamps)
    parts = mask_windows(data.timestamps, boundaries, horizon=int(config["horizon_bars"]))
    scored = fit_scaler(data, parts["train"])  # keep the training scaler

    emitted, skipped = 0, 0
    existing = {r["advisory_id"] for r in ledger.read() if r.get("kind") == "advisory"}
    for pid in sorted(set(scored.product_ids.tolist())):
        rows = np.flatnonzero(
            (scored.product_ids == pid)
            & (scored.position_in_asset >= int(config["sequence_length"]) - 1)
        )
        if not len(rows):
            continue
        row = int(rows[-1])
        windows, _labels, _net = make_batch(
            scored, np.array([row]), int(config["sequence_length"])
        )
        with torch.no_grad():
            state_taps, option_taps = encoder(windows)
            logits = head(
                state_taps,
                torch.ones(1, windows.shape[1]),
                option_taps,
                torch.ones(1, len(OPTIONS)),
            )
            probs = torch.softmax(logits, dim=-1)[0].numpy()
        asof = str(pd.Timestamp(scored.timestamps[row]))
        aid = advisory_id(pid, asof)
        if aid in existing:
            skipped += 1
            continue
        advisory = TradingAdvisory(
            product_id=pid,
            asof=asof,
            schema_version=1,
            options=OPTIONS,
            probabilities={name: float(p) for name, p in zip(OPTIONS, probs, strict=True)},
            top_option=OPTIONS[int(np.argmax(probs))],
            top_probability=float(np.max(probs)),
            model_sha256=digest,
            feature_schema=tuple(columns),
            horizon_bars=int(config["horizon_bars"]),
            round_trip_bps=float(config["round_trip_bps"]),
            validation_state=args.validation_state,
            metadata={
                "note": "advisory only; no order, size or venue is expressed",
                "run": str(args.run),
            },
        )
        ledger.emit(advisory, aid)
        emitted += 1
        print(
            f"{pid:10} {asof}  {advisory.top_option:>4} "
            f"p={advisory.top_probability:.3f}  -> {aid}"
        )
    print(f"\nemitted {emitted}, skipped {skipped} already present")
    print(f"ledger: {args.ledger}")
    print(
        "authority: dispatch_allowed=false, approval_granted=false — "
        "consumers must treat this as an opinion, not an instruction"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())