#!/usr/bin/env python3
"""Validate the trading decision head: walk-forward, non-overlapping, with CIs.

Retrains the head on each walk-forward fold's expanding training window and
evaluates on the held-out period after it, so no number here comes from a model
that saw the period it is scored on.

Two things are deliberately made harder than in the training script:

* **Positions are thinned to non-overlapping.** The label spans ``horizon`` bars,
  so consecutive hourly rows are overlapping claims on the same future. Scoring
  all of them treats ~168 correlated positions per asset as independent, which
  inflates Sharpe.
* **The skill term gets a bootstrap CI.** ``mean(s*r)`` contains a
  ``size x average return`` term that is not prediction. Only ``cov(s, r)`` is
  evidence, and the interval says whether it separates from zero.

Usage:
    python scripts/validate_trading_model.py --horizon 168 --folds 4 --per-asset
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from my_jev.decision_attention import ExternalDecisionAttentionHead  # noqa: E402
from my_jev.trading import OPTIONS  # noqa: E402
from my_jev.trading.dataset import LabelConfig, load_universe  # noqa: E402
from my_jev.trading.encoder import MarketTapEncoder  # noqa: E402
from my_jev.trading.validation import (  # noqa: E402
    Fold,
    bootstrap_skill_ci,
    non_overlapping,
    per_asset_breakdown,
    skill_and_pnl,
    walk_forward,
)
from my_jev.trading.windows import (  # noqa: E402
    fit_scaler,
    make_batch,
    stack_series,
    warmed_up,
)
from train_trading_head import _subsample  # noqa: E402

GO_CRITERIA = (
    "skill CI lower bound above zero in a majority of folds, aggregate "
    "Sharpe > 0.5, and no single asset carrying most of the skill"
)


def predict(
    encoder: nn.Module,
    head: nn.Module,
    data,
    index: np.ndarray,
    sequence_length: int,
    *,
    cost: float,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    head.eval()
    signed, raw = [], []
    with torch.no_grad():
        for start in range(0, len(index), batch_size):
            rows = index[start : start + batch_size]
            windows, _labels, net = make_batch(data, rows, sequence_length)
            state_taps, option_taps = encoder(windows)
            logits = head(
                state_taps,
                torch.ones(windows.shape[0], windows.shape[1]),
                option_taps,
                torch.ones(windows.shape[0], len(OPTIONS)),
            )
            probs = torch.softmax(logits, dim=-1).numpy()
            signed.append(probs[:, OPTIONS.index("buy")] - probs[:, OPTIONS.index("sell")])
            raw.append(net.numpy() + cost)  # net -> raw
    return np.concatenate(signed), np.concatenate(raw)


def train_fold(
    data,
    train_idx: np.ndarray,
    *,
    feature_count: int,
    hidden_size: int,
    depth: int,
    sequence_length: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
):
    torch.manual_seed(seed)
    encoder = MarketTapEncoder(
        feature_count=feature_count, hidden_size=hidden_size, depth=depth
    )
    head = ExternalDecisionAttentionHead(hidden_size=hidden_size, tap_count=4, rank=64)
    optimiser = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()), lr=lr
    )
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    for _epoch in range(epochs):
        encoder.train()
        head.train()
        order = rng.permutation(train_idx)
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            windows, labels, _net = make_batch(data, rows, sequence_length)
            state_taps, option_taps = encoder(windows)
            logits = head(
                state_taps,
                torch.ones(windows.shape[0], windows.shape[1]),
                option_taps,
                torch.ones(windows.shape[0], len(OPTIONS)),
            )
            loss = loss_fn(logits, labels)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()), 1.0
            )
            optimiser.step()
    return encoder, head


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/trading"))
    parser.add_argument("--horizon", type=int, default=168)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--round-trip-bps", type=float, default=40.0)
    parser.add_argument("--max-records", type=int, default=9000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-asset", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/validation"))
    args = parser.parse_args()

    config = LabelConfig(horizon_bars=args.horizon, round_trip_bps=args.round_trip_bps)
    series = _subsample(load_universe(args.root, config=config), args.max_records)
    raw_data = stack_series(series, sequence_length=args.sequence_length)
    folds = walk_forward(raw_data.timestamps, n_folds=args.folds, horizon=args.horizon)
    spacing = pd.Timedelta(hours=args.horizon)
    # Rows too early in an asset to form a full window cannot be trained or
    # scored on. The buffer keeps them (a window needs its history); they are
    # excluded from the fold indices here.
    usable = np.zeros(len(raw_data.labels), dtype=bool)
    usable[warmed_up(raw_data, args.sequence_length)] = True
    folds = [
        Fold(
            index=f.index,
            train=f.train[usable[f.train]],
            test=f.test[usable[f.test]],
            train_end=f.train_end,
            test_start=f.test_start,
            test_end=f.test_end,
        )
        for f in folds
    ]

    print(
        f"H={args.horizon} ({args.horizon / 24:.1f}d)  cost={args.round_trip_bps:.0f}bp  "
        f"records={len(raw_data.labels)}  folds={len(folds)}  spacing={spacing}",
        flush=True,
    )
    for fold in folds:
        print(
            f"  fold {fold.index}: train={len(fold.train):>6} test={len(fold.test):>6} "
            f"{fold.test_start} -> {fold.test_end}",
            flush=True,
        )

    signed_parts, raw_parts, index_parts = [], [], []
    per_fold = []
    for fold in folds:
        started = time.time()
        # Scaler refit per fold on that fold's training rows only.
        data = fit_scaler(raw_data, fold.train)
        encoder, head = train_fold(
            data,
            fold.train,
            feature_count=len(data.columns),
            hidden_size=args.hidden_size,
            depth=args.depth,
            sequence_length=args.sequence_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + fold.index,
        )
        thin = non_overlapping(
            fold.test,
            product_ids=data.product_ids,
            timestamps=data.timestamps,
            spacing=spacing,
        )
        signed, raws = predict(
            encoder, head, data, thin, args.sequence_length, cost=config.cost
        )
        metrics = skill_and_pnl(signed, raws, cost_bps=args.round_trip_bps)
        ci = bootstrap_skill_ci(signed, raws, seed=args.seed)
        metrics.update(
            {
                "skill_ci_low": ci["low"],
                "skill_ci_high": ci["high"],
                "skill_p_positive": ci["p_positive"],
                "fold": fold.index,
                "test_rows": int(len(fold.test)),
                "non_overlapping_rows": int(len(thin)),
                "seconds": round(time.time() - started, 1),
            }
        )
        per_fold.append(metrics)
        print(
            f"  fold {fold.index}: thinned {len(fold.test)}->{len(thin)}  "
            f"n_active={metrics.get('n_active', 0):>5} "
            f"hit={metrics.get('hit_rate', float('nan')):.3f} "
            f"skill={metrics.get('skill_bp', float('nan')):+8.2f}bp "
            f"CI[{metrics['skill_ci_low']:+.1f},{metrics['skill_ci_high']:+.1f}] "
            f"net={metrics.get('net_bp', float('nan')):+8.2f}bp "
            f"sharpe={metrics.get('sharpe', float('nan')):+.3f} "
            f"({metrics['seconds']}s)",
            flush=True,
        )
        signed_parts.append(signed)
        raw_parts.append(raws)
        index_parts.append(thin)

    signed = np.concatenate(signed_parts)
    raws = np.concatenate(raw_parts)
    index = np.concatenate(index_parts)
    aggregate = skill_and_pnl(signed, raws, cost_bps=args.round_trip_bps)
    ci = bootstrap_skill_ci(signed, raws, n_resamples=4000, seed=args.seed)

    concentration: dict[str, float] = {}
    if args.per_asset:
        breakdown = per_asset_breakdown(
            index, signed, raws, raw_data.product_ids, cost_bps=args.round_trip_bps
        )
        # Share of *absolute* skill, so one asset cannot hide behind a cancelling
        # negative elsewhere.
        total = sum(abs(v.get("skill_bp", 0.0)) for v in breakdown.values()) or 1.0
        concentration = {
            pid: v.get("skill_bp", 0.0) / total for pid, v in breakdown.items()
        }
        print()
        print(f"{'asset':10} {'n':>6} {'hit':>7} {'skill_bp':>10} {'net_bp':>9} {'share':>7}")
        for pid, values in breakdown.items():
            print(
                f"{pid:10} {values.get('n_active', 0):6} "
                f"{values.get('hit_rate', float('nan')):7.3f} "
                f"{values.get('skill_bp', float('nan')):+10.2f} "
                f"{values.get('net_bp', float('nan')):+9.2f} "
                f"{concentration[pid] * 100:6.1f}%"
            )

    positives = sum(1 for m in per_fold if m.get("skill_ci_low", float("nan")) > 0)
    sharpe = aggregate.get("sharpe", float("nan"))
    sharpe_ok = bool(sharpe == sharpe and sharpe > 0.5)
    concentration_ok = (
        max(concentration.values()) < 0.5 if concentration else True
    )
    go = bool(positives > len(per_fold) / 2 and sharpe_ok and concentration_ok)

    print()
    print("AGGREGATE (walk-forward, non-overlapping positions):")
    for key in ("n_active", "hit_rate", "mean_size", "skill_bp", "net_bp", "sharpe"):
        print(f"  {key:>12}: {aggregate.get(key, float('nan')):.4f}")
    print(
        f"  skill CI95  : [{ci['low']:+.2f}, {ci['high']:+.2f}] bp   "
        f"P(skill>0)={ci['p_positive']:.3f}"
    )
    print()
    print(f"folds with skill CI above zero : {positives}/{len(per_fold)}")
    print(f"aggregate Sharpe               : {sharpe:+.3f}")
    if concentration:
        worst = max(concentration, key=lambda k: concentration[k])
        print(f"largest single-asset skill share: {worst} {concentration[worst] * 100:.0f}%")
    print()
    print(f"VERDICT: {'GO' if go else 'NO-GO'}   (criteria: {GO_CRITERIA})")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"validation_h{args.horizon}.json").write_text(
        json.dumps(
            {
                "config": {
                    **config.as_metadata(),
                    "folds": args.folds,
                    "epochs": args.epochs,
                    "sequence_length": args.sequence_length,
                    "max_records": args.max_records,
                    "seed": args.seed,
                },
                "per_fold": per_fold,
                "aggregate": aggregate,
                "skill_ci": ci,
                "per_asset_share": concentration,
                "verdict": {
                    "folds_with_positive_skill_ci": positives,
                    "aggregate_sharpe": sharpe,
                    "sharpe_over_half": sharpe_ok,
                    "concentration_ok": concentration_ok,
                    "GO": go,
                },
            },
            indent=2,
        )
    )
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())