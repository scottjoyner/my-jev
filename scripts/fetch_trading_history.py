#!/usr/bin/env python3
"""Fetch multi-year Coinbase candle history for the trading lane.

The fetcher is incremental: it only requests the gap between the newest cached
bar and the requested end, so re-running after a partial run resumes rather than
restarting. That matters because multi-year hourly history is thousands of
paged requests per asset.

Usage:
    python scripts/fetch_trading_history.py --root data/trading --years 2
    python scripts/fetch_trading_history.py --root data/trading --granularity 86400
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from my_jev.trading.candles import (  # noqa: E402
    CandleFetchError,
    CandleRequest,
    cache_path,
    update_cache,
)

# Liquid spot pairs. Deliberately a short, high-volume list: breadth is worth
# less than an unbroken history per asset, because a gap silently shortens the
# usable sample for that asset only.
DEFAULT_UNIVERSE = (
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "XRP-USD",
    "DOGE-USD",
    "ADA-USD",
    "LINK-USD",
    "AVAX-USD",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/trading"))
    parser.add_argument("--universe", nargs="*", default=list(DEFAULT_UNIVERSE))
    parser.add_argument("--granularity", type=int, default=3600)
    parser.add_argument("--years", type=float, default=2.0)
    parser.add_argument("--end", type=str, default="")
    args = parser.parse_args()

    end = (
        datetime.fromisoformat(args.end).replace(tzinfo=UTC)
        if args.end
        else datetime.now(tz=UTC)
    )
    start = end - timedelta(days=int(365.25 * args.years))

    failures = 0
    for product_id in args.universe:
        target = cache_path(args.root, product_id, args.granularity)
        request = CandleRequest(product_id, args.granularity, start, end)
        try:
            frame = update_cache(args.root, request)
        except CandleFetchError as exc:
            failures += 1
            print(f"[FAIL] {product_id}: {exc}", flush=True)
            continue
        span_days = (
            (frame["timestamp"].max() - frame["timestamp"].min()).total_seconds() / 86400
            if len(frame) > 1
            else 0.0
        )
        gaps = int((frame["timestamp"].diff().dt.total_seconds() > args.granularity).sum())
        print(
            f"[ok] {product_id:10} rows={len(frame):7} span={span_days:7.1f}d gaps={gaps:4} "
            f"-> {target}",
            flush=True,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())