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

## The label tolerance is a second-order lever (and the metric that looked like a first-order one was circular)

`tie_ratio` decides which options count as equally good, and the system is
scored on regret. Four bundles from identical evidence, differing only in label
softness, 32 epochs, 3 seeds, best checkpoint per seed:

| tie_ratio | train records with >1 near-best option | mean regret ms/record | per seed |
|---|---|---|---|
| 1.00 | 0 / 152 (hard labels) | 51.7 | 51.2, 51.8, 52.0 |
| 1.03 (default) | 19 / 152 | 54.4 | 49.4, 47.7, 66.2 |
| 1.05 | 42 / 152 | 49.3 | 46.5, 52.2 |
| 1.10 | 84 / 152 | 47.4 | 46.1, 46.5, 49.8 |

The spread across tie ratios (47-54 ms) is smaller than the spread *within* an
arm (1.03 spans 47.7-66.2), so this does not resolve a winner and the default
stays at 1.03. The 1.00 and 1.10 arms are notably tighter across seeds than
1.03, which is suggestive but not enough to act on.

What the sweep did surface is a defect in how this was measured.
`near_best_hit_rate` is computed from `near_best_signature_ids`, which the
producer builds with `tie_ratio` — so the headline rate moved from 0.70 to 0.93
as the tie widened, with the router doing exactly the same thing. Comparing hit
rates across label policies is circular. `exact_best_hit_rate` (PR #10) reads
the frozen `wall_ms` and is tie-independent; regret always was.

**So the honest ranking of levers so far**, strongest first: training budget
(2x), leak-free splits, distinct specs per record, and measurability itself
(determinism, multi-seed). Label softness is not on that list, and neither is a
predicted-length feature. Both were plausible and both measured flat.

## What the model has to beat (and how to keep the comparison honest)

Trivial routers are the baseline a learned router must clear. Same 30 held-out
records, same three options, same oracle:

| policy | regret ms/record | ratio |
|---|---|---|
| always `none` | 751.8 | 1.5955 |
| always `dflash` | 186.5 | 1.1477 |
| per-family rule, fitted on train | 177.1 | 1.1403 |
| always `mtp` | 102.6 | 1.0812 |
| **trained model, 32 epochs, 3 seeds** | **57.4** | **1.0345** |
| oracle | 0 | 1.0000 |

Two mistakes this table exists to prevent:

- **Comparing a learned router against a constant while it is undertrained.**
  At 3 epochs the model scores 1.116 and *loses* to `always mtp` at 1.081 — which
  reads as "the state carries no signal, just ship the constant". At 32 epochs
  it wins by 44%. The honest reading is that the constant is a floor the model
  must clear, and a model below it has not been trained, not that the decision
  is unlearnable.
- **Fitting a rule on train and reading its held-out number as a target.** A
  per-family mode rule looks good on the full target leg (8.5 s vs 11.9 s) and
  loses to `always mtp` on the held-out split (177.1 vs 102.6). Rules fitted on
  train have to be scored on test like anything else.

## Where the remaining regret lives

Per-cell analysis over all 256 case x context cells:

- **The labels are reproducible.** Repeating 8 cases three times each: the
  per-case winner held in 7 of 8 (the eighth had spread > difference). Median
  signal 214 ms against a median within-policy spread of 42 ms. These are not
  coin flips.
- **`completion_tokens` explains the winner, and it is post-hoc.** dflash wins
  32/34 cells at >= 64 completion tokens (94%) and 113/222 (51%) below 64. A
  post-hoc length rule cuts corpus regret from 33.7 s (always-mtp) to 13.9 s.
- **Pre-execution observables see very little of it.** The best prompt-length
  threshold recovers 6% of the always-mtp-to-oracle gap; the best train-fitted
  per-family rule does not beat `always mtp` on held-out data.
- **The model beats both anyway** (57.4 vs 102.6 ms/record), so it is finding
  signal that neither feature exposes.

That points at the next lever: give the state something the model currently has
to infer — predicted output length, or per-family throughput history — and
re-measure. The remaining regret is concentrated (4 of 30 records carry 80% of
it) rather than spread evenly.

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