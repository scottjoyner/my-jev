#!/usr/bin/env python3
"""Compare horizon sweep runs and state plainly where the model has an edge.

Reads every ``report.json`` under a directory and ranks the horizons by realised
edge per unit of exposure. The point is to avoid reading a single configuration's
sign as a verdict: a model with a real but small directional tilt is unprofitable
at a 4-hour horizon and profitable at a weekly one purely because of what the
round trip costs relative to the move.

Usage:
    python scripts/compare_trading_runs.py runs/sweep
    python scripts/compare_trading_runs.py runs/sweep --cost-bps 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_reports(root: Path) -> list[dict]:
    reports = []
    for path in sorted(root.rglob("report.json")):
        try:
            reports.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return reports


def break_even_hit_rate(cost_bps: float, mean_abs_move_bp: float) -> float:
    """Hit rate at which a directional call exactly covers the round trip."""
    cost = cost_bps / 10_000.0
    move = mean_abs_move_bp / 10_000.0
    if move <= 0:
        return float("nan")
    return cost / (move + cost)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("runs/sweep"))
    parser.add_argument("--cost-bps", type=float, default=40.0)
    args = parser.parse_args()

    reports = load_reports(args.root)
    if not reports:
        print(f"no report.json under {args.root}")
        return 1

    rows = []
    for report in reports:
        config = report.get("config", {})
        head = report.get("test", {}).get("head", {})
        if not head:
            continue
        rows.append(
            {
                "horizon": config.get("horizon_bars"),
                "seq": config.get("sequence_length"),
                "cost": config.get("round_trip_bps", args.cost_bps),
                "acc": head.get("accuracy", float("nan")),
                "f1": head.get("macro_f1", float("nan")),
                "hit": head.get("hit_rate", float("nan")),
                "size": head.get("mean_size_when_active", 0.0),
                "edge": head.get("edge_bp_before_cost", float("nan")),
                "net": head.get("net_bp_when_active", float("nan")),
                "active": head.get("active_fraction", 0.0),
            }
        )
    if not rows:
        print("reports contained no test.head block")
        return 1
    rows.sort(key=lambda r: (r["horizon"] is None, r["horizon"]))

    print(
        f"{'H':>5} {'seq':>5} {'acc':>7} {'f1':>7} {'hit':>7} {'size':>6} "
        f"{'edge/cost':>11} {'net':>9} {'active':>8} {'profitable':>11}"
    )
    for row in rows:
        edge, cost = row["edge"], row["cost"]
        ratio = edge / cost if cost else float("nan")
        profitable = "yes" if row["net"] > 0 else "no"
        marker = "  <-- beats cost" if row["net"] > 0 else ""
        print(
            f"{row['horizon']:>5} {row['seq']:>5} {row['acc']:7.4f} {row['f1']:7.4f} "
            f"{row['hit']:7.3f} {row['size']:6.2f} {ratio:11.3f} {row['net']:+9.2f} "
            f"{row['active'] * 100:7.1f}% {profitable:>11}{marker}"
        )

    best = max(rows, key=lambda r: r["net"])
    print()
    print(
        f"best net: H={best['horizon']} ({best['net']:+.2f}bp/trade, "
        f"hit {best['hit']:.3f}, edge {best['edge']:+.2f}bp before cost)"
    )
    if best["net"] <= 0:
        print(
            "no horizon covers its round trip. Directional edge exists "
            f"({best['edge']:+.2f}bp) but is smaller than the "
            f"{best['cost']:.0f}bp cost."
        )
        print("Levers, in order of expected effect:")
        print("  1. longer horizon (cost amortised over a larger move)")
        print("  2. cheaper execution (maker fills, wider horizon still)")
        print("  3. per-asset models, so drift is not pooled into the label")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())