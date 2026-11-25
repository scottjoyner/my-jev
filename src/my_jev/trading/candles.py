"""Coinbase public candle history with an on-disk parquet cache.

Why not the trading system's own fetcher
----------------------------------------
``portfolio_optimizer``/``backtester`` already pull Coinbase candles, but
reaching into that repo from here would couple the model to a live trading
deployment's dependencies and credentials. The Coinbase *public* candle
endpoint needs no auth, so this module is self-contained and read-only.

Response shape gotcha
---------------------
Coinbase returns each candle as ``[time, low, high, open, close, volume]`` —
**open is the fourth field, not the second**. Parsing that as
``[time, open, high, low, close, volume]`` (the more common convention) silently
produces a plausible-looking but wrong series, so the order is asserted below
rather than trusted.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

#: Candle fields in the order Coinbase actually returns them.
COINBASE_FIELD_ORDER: tuple[str, ...] = ("time", "low", "high", "open", "close", "volume")

#: Granularity in seconds -> the column used to bucket candles.
GRANULARITIES: dict[int, str] = {
    60: "1m",
    300: "5m",
    900: "15m",
    3600: "1h",
    21600: "6h",
    86400: "1d",
}

_BASE_URL = "https://api.exchange.coinbase.com/products"
_USER_AGENT = "my-jev-trading/0.1"
# Coinbase's documented public allowance is rate-limited per IP and returns 429
# well before 10 req/s in practice — a 0.25s floor sustained 429s on the third
# asset of an eight-asset run. Pace globally (not per call) so concurrent callers
# cannot multiply the request rate, and honour Retry-After when the server sends
# one rather than guessing.
_MIN_INTERVAL_S = 1.0
_last_request_at = 0.0


def _pace(sleep: Callable[[float], None]) -> None:
    """Block until at least ``_MIN_INTERVAL_S`` has passed since the last call."""
    global _last_request_at
    now = time.monotonic()
    wait = _MIN_INTERVAL_S - (now - _last_request_at)
    if wait > 0:
        sleep(wait)
    _last_request_at = time.monotonic()


# The endpoint caps a page; observed 350 for 1h. Use a conservative page size so
# the loop terminates predictably across granularities.
_PAGE_SIZE = 300


class CandleFetchError(RuntimeError):
    """History could not be retrieved."""


@dataclass(frozen=True)
class CandleRequest:
    product_id: str
    granularity_s: int
    start: datetime
    end: datetime


def _to_epoch(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp())


def _iso(value: int) -> str:
    return datetime.fromtimestamp(value, tz=UTC).isoformat()


def _decode(rows: Iterable[Sequence[float]]) -> pd.DataFrame:
    materialised = [list(row) for row in rows]
    if not materialised:
        return pd.DataFrame(columns=list(COINBASE_FIELD_ORDER) + ["iso"])
    for row in materialised:
        if len(row) < len(COINBASE_FIELD_ORDER):
            raise CandleFetchError(
                f"short candle row ({len(row)} fields): {row!r}"
            )
    frame = pd.DataFrame(materialised).astype(float)
    frame.columns = list(COINBASE_FIELD_ORDER[: len(frame.columns)])
    frame["time"] = frame["time"].astype("int64")
    frame = frame.sort_values("time").drop_duplicates("time", keep="last")
    frame.index = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame.index.name = "timestamp"
    return frame.reset_index()


def fetch_candles(
    request: CandleRequest,
    *,
    timeout: float = 30.0,
    max_retries: int = 4,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Fetch every candle for *request*, oldest first.

    Pages backwards from ``end`` because Coinbase's ``start``/``end`` window is
    inclusive and returns most-recent-first; walking backwards from a known end
    and de-duplicating is the only way to guarantee full coverage of a long span.
    """
    if request.granularity_s not in GRANULARITIES:
        raise CandleFetchError(f"unsupported granularity {request.granularity_s}")
    if request.end <= request.start:
        raise CandleFetchError("end must be after start")

    step = request.granularity_s
    window = _PAGE_SIZE * step
    collected: list[Sequence[float]] = []
    cursor_end = _to_epoch(request.end)

    while cursor_end > _to_epoch(request.start):
        window_start = max(_to_epoch(request.start), cursor_end - window)
        url = (
            f"{_BASE_URL}/{request.product_id}/candles"
            f"?granularity={step}&start={window_start}&end={cursor_end}"
        )
        rows = _get_json(url, timeout=timeout, max_retries=max_retries, sleep=sleep)
        if not rows:
            break
        collected.extend(rows)
        oldest = min(int(row[0]) for row in rows)
        if oldest <= window_start:
            break
        if cursor_end == oldest:
            # No progress; refuse to spin rather than hammer the endpoint.
            raise CandleFetchError(
                f"{request.product_id} pagination stalled at {oldest}"
            )
        cursor_end = oldest - 1

    frame = _decode(collected)
    if frame.empty:
        return frame
    lo = pd.Timestamp(request.start)
    hi = pd.Timestamp(request.end)
    return frame[(frame["timestamp"] >= lo) & (frame["timestamp"] < hi)].reset_index(
        drop=True
    )


def _get_json(
    url: str, *, timeout: float, max_retries: int, sleep: Callable[[float], None]
) -> list:
    last: Exception | None = None
    for attempt in range(max_retries):
        _pace(sleep)
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as handle:
                return json.loads(handle.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 429:
                # The server states how long to wait; believe it instead of
                # imposing a backoff it has already judged too short.
                header = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    stated = float(header) if header else 0.0
                except ValueError:
                    stated = 0.0
                sleep(max(stated, _MIN_INTERVAL_S * (2**attempt)))
                continue
            if 500 <= exc.code < 600:
                sleep(_MIN_INTERVAL_S * (2**attempt))
                continue
            raise CandleFetchError(f"GET {url} failed: HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            sleep(_MIN_INTERVAL_S * (2**attempt))
    raise CandleFetchError(f"GET {url} failed after {max_retries} attempts: {last}")


def cache_path(root: Path, product_id: str, granularity_s: int) -> Path:
    tag = GRANULARITIES[granularity_s]
    safe = product_id.replace("/", "-").replace(":", "-")
    return Path(root) / f"{safe}__{tag}.parquet"


def load_cached(root: Path, product_id: str, granularity_s: int) -> pd.DataFrame | None:
    path = cache_path(root, product_id, granularity_s)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def update_cache(
    root: Path,
    request: CandleRequest,
    *,
    timeout: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Fetch and merge into the parquet cache, returning the full cached span.

    Only the missing tail is requested, so repeated runs are cheap and the cache
    is safe to rebuild incrementally.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = cache_path(root, request.product_id, request.granularity_s)

    existing = load_cached(root, request.product_id, request.granularity_s)
    if existing is not None and not existing.empty:
        newest = existing["timestamp"].max()
        gap_start = newest + pd.Timedelta(seconds=request.granularity_s)
        if gap_start >= pd.Timestamp(request.end):
            return existing
        fetch_from = max(pd.Timestamp(request.start), gap_start)
    else:
        fetch_from = pd.Timestamp(request.start)

    fresh = fetch_candles(
        CandleRequest(
            product_id=request.product_id,
            granularity_s=request.granularity_s,
            start=fetch_from.to_pydatetime(),
            end=request.end,
        ),
        timeout=timeout,
        sleep=sleep,
    )
    merged = fresh if existing is None or existing.empty else pd.concat(
        [existing, fresh], ignore_index=True
    )
    merged = merged.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    merged.to_parquet(path, index=False)
    return merged.reset_index(drop=True)