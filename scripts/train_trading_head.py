#!/usr/bin/env python3
"""Train the trading decision-attention head and report it against baselines.

The baseline comparison is the point of this script. A 3-class model on 138k
records will produce a confident-looking accuracy number almost regardless of
whether the features carry signal, so the headline figure reported here is the
model's edge over two references:

* ``majority``  — always predict the most common training label.
* ``logistic``  — multinomial logistic regression on the *last bar's* features,
                  which is the honest question "does the attention head beat a
                  linear model on the same inputs?".

A model that cannot beat the logistic baseline has not earned the extra
parameters, and this script will say so.

Usage:
    python scripts/train_trading_head.py --epochs 3
    python scripts/train_trading_head.py --epochs 3 --per-asset-report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from my_jev.decision_attention import ExternalDecisionAttentionHead  # noqa: E402
from my_jev.trading import OPTIONS  # noqa: E402
from my_jev.trading.dataset import LabelConfig, load_universe  # noqa: E402
from my_jev.trading.encoder import MarketTapEncoder, count_parameters  # noqa: E402
from my_jev.trading.splits import calendar_boundaries, mask_windows  # noqa: E402
from my_jev.trading.windows import (  # noqa: E402
    fit_scaler,
    make_batch,
    stack_series,
    warmed_up,
)

ACTION_COST_BPS = 40.0


def classification_metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Accuracy, macro-F1 and log-loss over the option set."""
    pred = probs.argmax(axis=1)
    accuracy = float((pred == labels).mean()) if len(labels) else 0.0
    losses = []
    for name in OPTIONS:
        hit = labels == OPTIONS.index(name)
        count = int(hit.sum())
        if not count:
            continue
        p = probs[hit, OPTIONS.index(name)]
        losses.append(float(-np.log(np.clip(p, 1e-12, 1.0)).mean()))
    f1s = []
    for index in range(len(OPTIONS)):
        tp = int(((pred == index) & (labels == index)).sum())
        fp = int(((pred == index) & (labels != index)).sum())
        fn = int(((pred != index) & (labels == index)).sum())
        if tp == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1s.append(float(2 * precision * recall / (precision + recall)))
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "log_loss": float(np.mean(losses)) if losses else 0.0,
    }


def decision_edge(
    probs: np.ndarray, raw_forward: np.ndarray, *, cost_bps: float = ACTION_COST_BPS
) -> dict[str, float]:
    """What the probabilities are worth once acted on.

    Each row becomes a signed position ``s = P(buy) - P(sell)`` in [-1, 1], where
    ``|s|`` is the fraction of maximum notional at risk. P&L is then::

        net = s * raw_forward - |s| * cost

    Two things this gets right that an earlier version did not:

    * The cost is charged on ``|s|``, the size actually risked. Charging the full
      round trip regardless of position size makes a mildly-confident model look
      like it loses exactly the cost on every trade, which is what produced a
      uniform "-40bp" reading that said nothing about directional skill.
    * It takes the **raw** forward return, not the cost-net one. Net was already
      net of the round trip, so passing net and subtracting the cost again
      double-charged it.

    ``hit_rate`` only counts rows where a position was actually taken, so a model
    that abstains is not credited with the coin-flip 50%.
    """
    signed = probs[:, OPTIONS.index("buy")] - probs[:, OPTIONS.index("sell")]
    size = np.abs(signed)
    gross = signed * raw_forward
    net = gross - size * (cost_bps / 10_000.0)
    # "Active" means a non-trivial position was taken. A model that expresses
    # near-zero conviction everywhere is abstaining, not trading.
    active = size >= 0.10
    if not active.any():
        return {
            "mean_net_bp": 0.0,
            "total_net_bp": 0.0,
            "active_fraction": 0.0,
            "net_bp_when_active": 0.0,
            "mean_size_when_active": 0.0,
            "hit_rate": 0.0,
            "edge_bp_before_cost": 0.0,
        }
    direction = np.sign(signed)
    return {
        "mean_net_bp": float(net.mean() * 1e4),
        "total_net_bp": float(net.sum() * 1e4),
        "active_fraction": float(active.mean()),
        "net_bp_when_active": float(net[active].mean() * 1e4),
        "mean_size_when_active": float(size[active].mean()),
        "hit_rate": float((direction[active] == np.sign(raw_forward[active])).mean()),
        # Directional skill with the cost removed, so a positive number here and a
        # negative net can be compared directly.
        "edge_bp_before_cost": float(gross[active].mean() * 1e4),
    }


def majority_probs(train_labels: np.ndarray, n: int) -> np.ndarray:
    counts = np.bincount(train_labels, minlength=len(OPTIONS)).astype(float)
    prior = counts / counts.sum()
    return np.tile(prior, (n, 1))


def logistic_probs(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray
) -> np.ndarray:
    """Multinomial logistic regression on the final bar's standardised features."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return majority_probs(train_y, len(test_x))
    # `multi_class` was removed in scikit-learn 1.9; multinomial is now the
    # default for a >2-class problem, so the argument is simply dropped.
    model = LogisticRegression(max_iter=400)
    model.fit(train_x, train_y)
    out = model.predict_proba(test_x)
    full = np.zeros((len(test_x), len(OPTIONS)))
    for column, cls in enumerate(model.classes_):
        full[:, cls] = out[:, column]
    return full


def evaluate_head(
    encoder: MarketTapEncoder,
    head: ExternalDecisionAttentionHead,
    data,
    index: np.ndarray,
    sequence_length: int,
    *,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    head.eval()
    chunks, nets = [], []
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
            chunks.append(torch.softmax(logits, dim=-1).numpy())
            nets.append(net.numpy())
    return np.concatenate(chunks), np.concatenate(nets)


def _subsample(series, target: int):
    """Stride-subsample each series to roughly *target* rows, period-agnostic."""
    from dataclasses import replace

    out = []
    for item in series:
        n = len(item.features)
        if n <= target:
            out.append(item)
            continue
        # Ceiling, not floor: n // target is 1 whenever target is close to n,
        # which silently disables subsampling entirely.
        stride = max(1, -(-n // target))
        out.append(
            replace(
                item,
                features=item.features.iloc[::stride].reset_index(drop=True),
                labels=item.labels.iloc[::stride].reset_index(drop=True),
                forward_return=item.forward_return.iloc[::stride].reset_index(drop=True),
                net_forward_return=item.net_forward_return.iloc[::stride].reset_index(drop=True),
            )
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/trading"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--round-trip-bps", type=float, default=40.0)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="stride-subsample this many rows per asset (0 = all). Attention "
        "cost is quadratic in sequence length, so a long-horizon sweep over "
        "138k rows takes hours; sampling keeps every asset represented and cuts "
        "an epoch by an order of magnitude.",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/trading"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = LabelConfig(horizon_bars=args.horizon, round_trip_bps=args.round_trip_bps)
    series = load_universe(args.root, config=config)
    if args.max_records:
        # Stride-sample, never truncate: taking the first N rows would keep a
        # single market regime, which is exactly the confound a horizon sweep
        # must not introduce.
        series = _subsample(series, args.max_records)
    raw = stack_series(series, sequence_length=args.sequence_length)

    # Rows are stacked per asset    # Rows are stacked per asset, so the frame is not one ordered timeline.
    # Derive the cut points from the shared calendar, then classify every row by
    # its own timestamp. That keeps each asset contiguous (window extraction
    # depends on it) while still cutting all assets at the same instants.
    boundaries = calendar_boundaries(
        raw.timestamps,
        valid_fraction=args.valid_fraction,
        test_fraction=args.test_fraction,
    )
    parts = mask_windows(
        raw.timestamps, boundaries, horizon=config.horizon_bars
    )
    # The buffer keeps every asset's warm-up rows because a window needs the
    # history behind it; warm-up rows are excluded from the *split* only.
    usable = warmed_up(raw, args.sequence_length)
    usable_mask = np.zeros(len(raw.labels), dtype=bool)
    usable_mask[usable] = True
    parts = {name: rows[usable_mask[rows]] for name, rows in parts.items()}
    train_idx, valid_idx, test_idx = parts["train"], parts["valid"], parts["test"]
    split = type(
        "Split",
        (),
        {
            "sizes": lambda self: {k: len(v) for k, v in parts.items()},
            "summary": lambda self: {
                **{k: len(v) for k, v in parts.items()},
                "valid_from": str(boundaries[0]),
                "test_from": str(boundaries[1]),
            },
        },
    )()
    data = fit_scaler(raw, train_idx)

    encoder = MarketTapEncoder(
        feature_count=len(data.columns),
        hidden_size=args.hidden_size,
        depth=args.depth,
    )
    head = ExternalDecisionAttentionHead(
        hidden_size=args.hidden_size, tap_count=4, rank=64
    )
    params = count_parameters(encoder) + count_parameters(head)
    print(
        f"records={len(data)} features={len(data.columns)} "
        f"params={params:,} split={split.sizes()}",
        flush=True,
    )

    optimiser = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()), lr=args.lr
    )
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(args.epochs):
        encoder.train()
        head.train()
        started = time.time()
        order = np.random.permutation(train_idx)
        total, seen = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            rows = order[start : start + args.batch_size]
            windows, labels, _net = make_batch(data, rows, args.sequence_length)
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
            total += loss.detach().item() * len(rows)
            seen += len(rows)
        valid_probs, _unused = evaluate_head(
            encoder, head, data, valid_idx, args.sequence_length
        )
        valid_metrics = classification_metrics(valid_probs, data.labels[valid_idx])
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / max(seen, 1),
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_macro_f1": valid_metrics["macro_f1"],
                "seconds": round(time.time() - started, 1),
            }
        )
        print(
            f"epoch {epoch}: loss={total / max(seen, 1):.4f} "
            f"valid_acc={valid_metrics['accuracy']:.4f} "
            f"valid_f1={valid_metrics['macro_f1']:.4f} "
            f"({history[-1]['seconds']}s)",
            flush=True,
        )

    test_probs, test_raw = evaluate_head(
        encoder, head, data, test_idx, args.sequence_length
    )
    test_labels = data.labels[test_idx]
    test_raw_forward = data.net_forward[test_idx] + config.cost  # undo the netting

    majority = majority_probs(data.labels[train_idx], len(test_idx))
    # The logistic baseline sees the *current bar's* features: a window ending at
    # row i ends at row i, so that is data.features[i]. (Indexing [:, -1] here
    # would silently take the last feature column, not the last bar.)
    logistic = logistic_probs(
        data.features[train_idx], data.labels[train_idx], data.features[test_idx]
    )

    report = {
        "config": {
            **config.as_metadata(),
            "sequence_length": args.sequence_length,
            "hidden_size": args.hidden_size,
            "depth": args.depth,
            "epochs": args.epochs,
            "seed": args.seed,
        },
        "split": split.summary(),
        "params": params,
        "history": history,
        "test": {
            "head": {
                **classification_metrics(test_probs, test_labels),
                **decision_edge(test_probs, test_raw_forward, cost_bps=config.round_trip_bps),
            },
            "majority": {
                **classification_metrics(majority, test_labels),
                **decision_edge(majority, test_raw_forward, cost_bps=config.round_trip_bps),
            },
            "logistic": {
                **classification_metrics(logistic, test_labels),
                **decision_edge(logistic, test_raw_forward, cost_bps=config.round_trip_bps),
            },
        },
    }
    head_block = report["test"]["head"]
    beats_logistic = head_block["accuracy"] > report["test"]["logistic"]["accuracy"]
    report["verdict"] = {
        "beats_logistic_on_accuracy": bool(beats_logistic),
        "beats_majority_on_accuracy": bool(
            head_block["accuracy"] > report["test"]["majority"]["accuracy"]
        ),
        "head_net_bp_when_active": head_block["net_bp_when_active"],
        "logistic_net_bp_when_active": report["test"]["logistic"]["net_bp_when_active"],
        "profitable_after_cost": bool(head_block["net_bp_when_active"] > 0.0),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "head": head.state_dict(),
            "config": report["config"],
            "columns": list(data.columns),
        },
        args.output / "model.pt",
    )

    print()
    for name in ("majority", "logistic", "head"):
        block = report["test"][name]
        print(
            f"{name:9} acc={block['accuracy']:.4f} macro_f1={block['macro_f1']:.4f} "
            f"logloss={block['log_loss']:.4f} "
            f"net_bp_active={block['net_bp_when_active']:+.1f} "
            f"active={block['active_fraction'] * 100:.1f}%"
        )
    print()
    print(f"beats logistic on accuracy : {report['verdict']['beats_logistic_on_accuracy']}")
    print(f"profitable after cost      : {report['verdict']['profitable_after_cost']}")
    print(f"wrote {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())