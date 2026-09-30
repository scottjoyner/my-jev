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

## Does more data help? (yes, and it has not saturated)

Only the training set size varies here. Validation, calibration and test are
the same leak-free records in every arm, so each arm is scored identically;
subsets are cut at whole task-evaluator-spec groups (12 / 24 / 47 specs) so
shrinking cannot put a held-out spec into training. 32 epochs, seeds 1-3,
deterministic.

| train records | specs | checkpoint | mean regret ms | range | ratio | near-best | p95 ms |
|---|---|---|---|---|---|---|---|
| 32 (25%) | 12 | best (nll) | 164.9 | 121.6-186.5 | 1.1355 | 0.57 | 950.7 |
| 32 (25%) | 12 | last | 161.8 | 119.0-246.2 | 1.1395 | 0.59 | 774.6 |
| 88 (50%) | 24 | best (nll) | 117.3 | 112.7-119.5 | 1.1089 | 0.66 | 595.2 |
| 88 (50%) | 24 | last | 124.2 | 116.1-137.8 | 1.1122 | 0.63 | 661.1 |
| 152 (100%) | 47 | best (nll) | 57.4 | 46.1-75.0 | 1.0345 | 0.76 | 219.3 |
| 152 (100%) | 47 | last | 60.8 | 43.8-72.4 | 1.0353 | 0.77 | 307.6 |

The seed ranges do not overlap between adjacent sizes (121.6-186.5 vs
112.7-119.5 vs 46.1-75.0), so this is a resolved effect and not noise: 4x the
records — and roughly 4x the distinct specs — roughly halves regret, lifts
near-best from 0.57 to 0.76, and pulls p95 from ~950 ms to ~220 ms. The curve is
still falling at 152 records, so the next rung up is worth its GPU hours.

An earlier reading of this data said "more rows of the same tasks did not help".
That was measured with a nondeterministic trainer, at 3 epochs, on a bundle
whose splits leaked. Every one of those three defects pushed in the direction of
the wrong answer.

## One nuisance parameter worth naming

Reproducibility means identical inputs give identical outputs; it does not mean
the output is invariant to how the inputs are *ordered*. Re-ordering the same
152 training records (same records, different file order) moved mean regret
from 68.2 ms to 57.4 ms — about 19%, comparable to the seed spread. Row order
is therefore a controlled variable in any comparison: hold the file fixed, and
change it deliberately rather than by accident.

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