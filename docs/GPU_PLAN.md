# Planning GPU time for a model version

`my-jev-gpu-plan` answers two questions before a run exists: which model version
is this job going to produce, and does that version deserve the GPU.

```bash
my-jev-gpu-plan spec.toml \
    --budget-gpu-minutes 1000 \
    --inventory local \
    --horizon-gpu-minutes 1000 \
    --output plan.json
```

Exit code is `0` only when the plan is **affordable and placeable**, so a shell
can gate without parsing the document. The plan is printed either way — "why is
this not actionable" is the useful output in the failing case.

## Planned versions

`registry.ExperimentEntry` records what happened: a run that finished, with the
SHAs of everything it produced. Nothing named the version that was *about to*
exist, while changing your mind was still cheap.

`version_id` is content-addressed over model, data, and training config. The
same job always plans to the same version; changing the job mints a new one
rather than silently amending the old.

The digest deliberately **excludes** bookkeeping — `output_root`, `registry_path`
— because moving a run directory does not change what gets trained, and
including them would mint a new version per directory. It **includes** the data
paths, so two jobs differing only in dataset do not collide.

## Readiness is asked first

`scale_readiness.evaluate_scale_readiness` already answers "is a small baseline
good enough to justify the full run", and documents itself as a
resource-allocation gate. The planner consults it rather than restating its
thresholds:

```bash
my-jev-gpu-plan spec.toml --budget-gpu-minutes 9999 --readiness-run-dir runs/x
```

When it does not pass, no training GPU is planned. The cheapest question gets
asked first, so a version nobody should fund never reaches budget arithmetic.

`readiness_from_run` maps an incomplete baseline to not-ready rather than letting
the evaluator's `ValueError` escape. An unfinished baseline is an ordinary
state, not a crash, and it means "not yet".

## The chain is all-or-nothing

```
train → benchmark → evaluate
```

Benchmarking and evaluating a version that was never trained produces no
evidence about the version. So a budget funding two of three stages funds
nothing:

```json
{"within_budget": false,
 "budget_reason": "the full train/benchmark/evaluate chain needs 892.8 GPU-minutes
                   but the budget is 500.0; a partial chain produces no evidence
                   about the version, so nothing is scheduled",
 "deferred_stages": ["train", "benchmark", "evaluate"]}
```

The stages are still listed with `gpu_ids: []`, so the cost breakdown that
explains the shortfall stays inspectable.

## Budget and placement are different questions

A plan can fit its budget and still have nowhere to run.

| field | question |
|---|---|
| `within_budget` | does the chain fit the minutes allowed? |
| `fully_placed` | did every stage land on at least one GPU? |

Conflating them would let a caller act on a plan that cannot be scheduled, so
the CLI requires both. When placement fails, the plan names the unplaced stages
and says the plan is not actionable until availability appears or a lease is
released.

## Inventories: measured vs assumed

`gpu_plan` allocates only from what it is told is free. Where that comes from
matters, so `gpu_inventory` keeps the two apart:

| fact | source | kind |
|---|---|---|
| which devices exist | PyTorch, else `nvidia-smi` | measurement |
| is a GPU busy right now | the `gpu` lease in `my_jev.locking` | measurement, `flock`-backed |
| how many minutes are free | `--horizon-gpu-minutes` | assumption |

Free minutes are an assumption and the module says so: nobody can know how long
a job will take before it runs. `--horizon-gpu-minutes` defaults to **0**, so the
safe behaviour is to plan nothing rather than invent availability.

The busy-or-free probe takes the same `ResourceRequest("gpu")` lease
`experiment.py` takes before it runs anything, so it cannot disagree with the run
it is planning for. It is a probe with a race in it — the lease is released
before the caller acts — which is acceptable because the plan is advisory and the
real run takes its own lease.

The horizon is **divided across devices** by default, because this repo's lease
model is a single exclusive `gpu` resource rather than one lease per card. Pass
`--horizon-is-per-device` to change that. For the same reason a plan placed
across several devices carries a note saying it contends rather than running
concurrently.

## Pickers

Each stage records which named rule resolved it:

| stage | picker | emits |
|---|---|---|
| `train` | `train_config` | lr, epochs, effective batch, seed |
| `benchmark` | `benchmark_suite` | batch size, whether fleet states are set |
| `evaluate` | `eval_thresholds` | the gates the version will be compared against |

Recorded by name so two plans can be compared rule-for-rule rather than only by
outcome. Rationales are composed from facts, not scraped from prose, so rewording
a message elsewhere cannot silently repoint them.

## Estimates, and stopping the estimate from inflating

`estimated_gpu_minutes` is a heuristic over the job's shape — epochs, effective
batch, sequence lengths, LoRA width, frozen backbone. It is labelled as an
estimate in `notes`, and each uncalibrated stage says so.

Supply `measured` from completed runs and it overrides the estimate for budgeting:

```bash
my-jev-gpu-plan spec.toml --budget-gpu-minutes 500 --measured measured.json
```

Without that, replanning after training keeps budgeting the full training
estimate for work already done, and the total inflates every time you re-plan.

## Advisory

`PlanAuthority` is all-false with `Literal[False]`, so supplying `True` is a
validation error. Nothing in `gpu_plan` or `gpu_inventory` acquires a lease to
keep — the probe releases before returning. Executing a plan means taking the
`LeaseSet` yourself, which is a decision this module deliberately does not make.
