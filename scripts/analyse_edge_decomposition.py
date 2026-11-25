#!/usr/bin/env python3
"""Decompose a run's reported edge into skill versus position-sizing artifact.

The sweep showed the decision head reporting ~20x the logistic baseline's edge at
a 7-day horizon while scoring a *worse* hit rate and holding ~4x the position
size. Those facts are only consistent if the edge is coming from concentrating
size where moves happen to be large, not from predicting direction better.

``edge_bp = mean(s * r)`` mixes two things:

    mean(s * r) = mean(s) * mean(r) + cov(s, r)

Only the second term is skill. The first is a bet size times an average return,
and it inflates whenever the model happens to be large in periods whose returns
are also large — which is exactly what a volatility-aware model does.

So report, for each model:

* ``hit_rate``          -- directional accuracy, size-independent
* ``edge_per_unit``     -- ``mean(s*r) / mean|s|`` -- skill per unit of exposure
* ``skill_bp``          -- ``cov(s, r)`` in bp -- skill with size effects removed
* ``sharpe``            -- ``mean(s*r) / std(s*r)`` -- edge per unit of realised
                           risk, the only figure comparable across sizes

Usage:
    python scripts/analyse_edge_decomposition.py runs/sweep/h168
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from my_jev.decision_attention import ExternalDecisionAttentionHead  # noqa: E402
from my_jev.trading import OPTIONS  # noqa: E402
from my_jev.trading.dataset import LabelConfig, load_universe  # noqa: E402
from my_jev.trading.encoder import MarketTapEncoder  # noqa: E402
from my_jev.trading.splits import calendar_boundaries, mask_windows  # noqa: E402
from my_jev.trading.windows import fit_scaler, make_batch, stack_series  # noqa: E402


def decompose(signed: np.ndarray, raw: np.ndarray, *, cost_bps: float) -> dict[str, float]:
    size = np.abs(signed)
    active = size >= 0.10
    if not active.any():
        return {}
    s = signed[active]
    r = raw[active]
    pnl = s * r - size[active] * (cost_bps / 10_000.0)
    # cov with ddof=0: a single deployment path, not a sampling estimate.
    skill = float(np.cov(s, r, ddof=0)[0, 1] * 1e4)
    beta = float(s.mean() * r.mean() * 1e4)
    return {
        "hit_rate": float((np.sign(s) == np.sign(r)).mean()),
        "mean_size": float(size[active].mean()),
        "edge_bp": float((s * r).mean() * 1e4),
        "edge_per_unit": float((s * r).mean() / size[active].mean() * 1e4),
        "size_x_return_bp": beta,
        "skill_bp_cov": skill,
        "net_bp": float(pnl.mean() * 1e4),
        "sharpe": float(pnl.mean() / (pnl.std(ddof=0) or 1e-12)),
        "active_fraction": float(active.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="a runs/<name> directory")
    parser.add_argument("--root", type=Path, default=Path("data/trading"))
    parser.add_argument("--max-records", type=int, default=12000)
    args = parser.parse_args()

    report = json.loads((args.run / "report.json").read_text())
    config = report["config"]
    horizon = int(config["horizon_bars"])
    cost_bps = float(config["round_trip_bps"])
    sequence = int(config["sequence_length"])

    label_config = LabelConfig(horizon_bars=horizon, round_trip_bps=cost_bps)
    series = load_universe(args.root, config=label_config)
    if args.max_records:
        from train_trading_head import _subsample

        series = _subsample(series, args.max_records)
    raw = stack_series(series, sequence_length=sequence)
    boundaries = calendar_boundaries(raw.timestamps)
    parts = mask_windows(raw.timestamps, boundaries, horizon=horizon)
    test = parts["test"]
    # The encoder was trained on train-fitted standardised features. Feeding it raw
    # features silently reproduces a different model (it reported a 0.38 hit rate
    # against the training run's 0.55), so the scaler must be rebuilt identically.
    data = fit_scaler(raw, parts["train"])

    blob = torch.load(args.run / "model.pt", map_location="cpu", weights_only=False)
    encoder = MarketTapEncoder(
        feature_count=len(blob["columns"]),
        hidden_size=int(config["hidden_size"]),
        depth=int(config["depth"]),
    ).eval()
    encoder.load_state_dict(blob["encoder"])

    # Reconstruct the *trained* decision, not a hand-rolled approximation: the
    # head's weights are in the checkpoint and its tap mixer is learned, so
    # averaging option vectors here would not reproduce the model's output.
    head = ExternalDecisionAttentionHead(
        hidden_size=int(config["hidden_size"]), tap_count=4, rank=64
    ).eval()
    head.load_state_dict(blob["head"])

    signed_chunks, raw_chunks = [], []
    with torch.no_grad():
        for start in range(0, len(test), 512):
            rows = test[start : start + 512]
            windows, _labels, _net = make_batch(data, rows, sequence)
            state_taps, option_taps = encoder(windows)
            logits = head(
                state_taps,
                torch.ones(windows.shape[0], windows.shape[1]),
                option_taps,
                torch.ones(windows.shape[0], len(OPTIONS)),
            )
            probs = torch.softmax(logits, dim=-1).numpy()
            signed_chunks.append(
                probs[:, OPTIONS.index("buy")] - probs[:, OPTIONS.index("sell")]
            )
            # raw forward return = net + cost
            raw_chunks.append(
                (data.net_forward[rows] + label_config.cost).astype(np.float64)
            )
    signed = np.concatenate(signed_chunks)
    raw = np.concatenate(raw_chunks)

    result = decompose(signed, raw, cost_bps=cost_bps)

    # Same decomposition for the linear baseline on the identical current-bar
    # features, so the comparison is like-for-like: same rows, same cost, same
    # metric. Comparing a head's decomposed edge against a baseline's raw edge
    # is exactly the mistake this script exists to prevent.
    baseline = {}
    try:
        from sklearn.linear_model import LogisticRegression

        train_rows = parts["train"]
        model = LogisticRegression(max_iter=500)
        model.fit(data.features[train_rows], data.labels[train_rows])
        probs = np.zeros((len(test), len(OPTIONS)))
        raw_probs = model.predict_proba(data.features[test])
        for column, cls in enumerate(model.classes_):
            probs[:, cls] = raw_probs[:, column]
        baseline = decompose(
            probs[:, OPTIONS.index("buy")] - probs[:, OPTIONS.index("sell")],
            raw,
            cost_bps=cost_bps,
        )
    except ImportError:
        pass
    print(f"run: {args.run}   H={horizon} ({horizon / 24:.1f}d)   cost={cost_bps:.0f}bp")
    print()
    keys = (
        "hit_rate",
        "mean_size",
        "edge_bp",
        "edge_per_unit",
        "size_x_return_bp",
        "skill_bp_cov",
        "net_bp",
        "sharpe",
        "active_fraction",
    )
    print(f"{'metric':>18} {'head':>12} {'logistic':>12}")
    for key in keys:
        head_value = result.get(key, float("nan"))
        base_value = baseline.get(key, float("nan"))
        print(f"{key:>18} {head_value:12.4f} {base_value:12.4f}")
    print()
    ratio = result.get("skill_bp_cov", 0.0) + result.get("size_x_return_bp", 0.0)
    if ratio:
        share = result["skill_bp_cov"] / ratio
        print(
            f"edge decomposition: {result['edge_bp']:.1f}bp = "
            f"{result['size_x_return_bp']:.1f}bp (size x average return) + "
            f"{result['skill_bp_cov']:.1f}bp (covariance = skill)"
        )
        print(f"  -> only {share * 100:.0f}% of the edge is directional skill")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())