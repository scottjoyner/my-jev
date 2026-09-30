# Training budget and how to compare two runs

## Why this exists

Two runs of the same command, same seed, same data, same 16 epochs:

```text
mean regret 58.2 ms (near-best 0.70)
mean regret 52.9 ms (near-best 0.73)
```

That was not a bug in the data or the model. `train.py` seeded `random`,
`numpy` and `torch`, and on CUDA that is not enough: reduction and scatter
kernels pick their accumulation order at runtime, so the same seed still lands
on a different model. Any comparison drawn from single runs — bundles, epoch
counts, checkpoint rules — was reading noise as wide as the effects.

`my_jev.determinism.configure_determinism(seed)` now pins the algorithms that
decide that order and records the outcome in `run_config.json`, so a run states
whether it was reproducible. Two seeded runs are bit-identical.

Run-to-run noise is gone. **Seed variance is not** — different seeds are
genuinely different models, and the spread below is real. Comparisons average
over seeds; they never use one.

## Measured budget

Leak-free bundle (256 records, 152/40/34/30 splits), seeds 1-3, deterministic,
evaluated on held-out test with regret against the frozen oracle:

| epochs | checkpoint | mean regret ms | ratio | near-best | p95 ms |
|---|---|---|---|---|---|
| 3 | best (nll) | 138.9 | 1.1297 | 0.60 | 700.3 |
| 3 | last | 120.6 | 1.1117 | 0.61 | 595.2 |
| 16 | best (nll) | 73.0 | 1.0569 | 0.73 | 285.0 |
| 16 | last | 86.3 | 1.0662 | 0.64 | 353.5 |
| 32 | best (nll) | 68.2 | 1.0395 | 0.74 | 397.9 |
| 32 | last | 57.8 | 1.0351 | 0.74 | 219.6 |

Read off it:

- **At this corpus size, the epoch budget is the dominant lever, not the
  corpus.** 3 → 32 epochs roughly halves mean regret and lifts near-best from
  0.60 to 0.74, and every seed improves. The 3-epoch default was costing about
  2x on this data.
- **More epochs also stabilise.** The seed range at 3 epochs is 119-178 ms; at
  32 it is 48-76 ms.
- **nll-based checkpoint selection is not reliably the right rule.** It wins
  at 16 epochs, loses at 3 and at 32, and averages worse than simply taking the
  final checkpoint (93.4 vs 88.2 ms) with a wider tail. Validation is 32
  records; selecting on nll optimises a proxy for the thing we are judged by.
  Both checkpoints are saved, so this is a choice to make per experiment, not a
  defect.

## Why the default stayed at 3

The 2x is measured on a 152-record training split. `TRAINING_PLAN.md` is right
that the expensive part at 200k+ states is diverse, verifiable decisions and
not epochs, and a global default of 32 would make every large run ten times
more expensive to try. So the default is unchanged and this table is the
evidence for passing `--epochs` explicitly.

## Comparing two bundles, properly

```bash
for seed in 1 2 3; do
  PYTHONPATH=src python3 -m my_jev.train --train $B/train.jsonl --valid $B/validation.jsonl \
    --output /tmp/$seed.pt --epochs 16 --seed $seed
  PYTHONPATH=src python3 -m my_jev.calibrate --checkpoint /tmp/$seed.pt/best \
    --data $B/calibration.jsonl --output /tmp/$seed.cal.json
  PYTHONPATH=src python3 -m my_jev.assistx_policy_eval --checkpoint /tmp/$seed.pt/best \
    --data $B/test.jsonl --calibration /tmp/$seed.cal.json --output /tmp/$seed.eval.json
done
```

Report the mean and the range over seeds. If the ranges overlap, the comparison
has not resolved anything and should not be written up as if it had.

Check the split for leakage before trusting any of it: cases sharing a task
evaluator spec must not straddle splits. `inference_policy_training_dataset` in
AssistX anchors duplicate specs on their first case id for exactly this reason.