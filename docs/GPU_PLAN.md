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

## Calibration against runs that finished

The heuristic above was 9x off against a real 100-minute run in testing, which is
the expected order of magnitude for "derived from the spec's shape". So the
planner can be calibrated against completed runs:

```bash
my-jev-gpu-plan spec.toml \
    --budget-gpu-minutes 500 \
    --calibration-registry runs/experiments/registry.json
```

`calibration_from_registry` reads `created_at`/`updated_at` from
`ExperimentRegistry` entries and derives a **scale** carrying the heuristic onto
measured reality.

### Only runs that completed the chain

| status | counted | why |
|---|---|---|
| `candidate`, `rejected` | yes | the chain ran to completion; the promotion verdict differs but the work happened |
| `failed` | no | stopped early, so its duration says nothing about a full chain |
| `dry_run` | no | never started |
| zero or negative span | no | a clock artefact, not a measurement |

Calibrating from a run that died in training would understate every subsequent
stage, which is the same failure as budgeting from a guess. Excluded runs are
named in the plan with their reason — a calibration that quietly dropped half its
samples would be unfalsifiable.

A registry with no completed run yields `None`, not a zero-sample calibration:
absent and empty mean different things, and only the second would justify
scheduling.

### Per-stage figures are measured, or clearly marked as inferred

`experiment._stage` writes `stages/<name>.timing.json` for every stage it runs,
recording start, finish, duration, and whether the stage completed. It writes
the file before the failure check, because a stage that died still consumed GPU
and losing that number is how the estimates rot in the first place.

```json
{"name": "train", "duration_seconds": 3600.4, "completed": true}
```

So a plan calibrated from an instrumented run uses **real** per-stage durations:

```bash
my-jev-gpu-plan spec.toml \
    --calibration-registry runs/experiments/registry.json \
    --calibration-run-dir runs/experiments/scale-probe-abc123
```

Runs from before that instrumentation existed still calibrate against the chain
total, with the split inferred from the estimator's shape. `per_stage_is_inferred`
says which happened, and it is **derived in a validator** rather than asserted —
a caller who sets it inconsistently with the data gets an error instead of a flag
that lies in whichever direction they chose.

Only completed stages count. A stage that failed burned real GPU, so its duration
is a real number, but it is not what the stage costs when it works.

## The stages are the ones the run really executes

An earlier version of this planner budgeted a `train → benchmark → evaluate`
chain. There is no `evaluate` stage in `experiment.py`; the pipeline runs
`train`, `calibrate`, `benchmark`, and — only when the job sets
`benchmark.fleet_states` — `fleet_benchmark`.

Because calibration reads the filenames `_stage` writes, a mismatch would not
just misforecast: it would silently discard every measurement. `StageKind` is now
the stage names the pipeline uses, and a test asserts each one appears in
`experiment.py`'s source.

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

## Executing a plan: device pinning

A plan that says a stage runs on `cuda:1` is worthless if nothing enforces it.
Every module here resolves `torch.device("cuda")` to whatever the visible set
allows, and **none of them reads an explicit index**. So `run_experiment` pins
`CUDA_VISIBLE_DEVICES` for the stage process:

```bash
my-jev-experiment spec.toml --gpu-plan plan.json
```

`stage_device_pin` translates `cuda:1` to `1`, and refuses a device this host does
not have visible — running it anyway would put the work somewhere the plan never
said. The pin is layered onto the current environment rather than replacing it,
since `subprocess.run` substitutes the whole environment when given `env`.

No changes were needed in the nine modules that pick a device, which is the
point of using the variable they already honour.

Rules:

- a stage the plan did not place runs **unpinned**, and records `device: null`;
  defaulting would put work on a device the plan never chose
- a plan naming a stage the pipeline will not run is **refused** — silently
  ignoring it would leave the operator believing it applied
- `--gpu-plan` is overridden by an explicit `device_assignments` mapping, so an
  operator can correct a plan without editing or regenerating it
- the intended pin is written to `stages/<name>.command.json`, so a run can be
  reproduced or diagnosed later without guessing which physical GPU it used
## Per-device leases, and why the migration needs no decision

Device pinning landed with the previous change, which is what made per-device
leases *meaningful*: a lease on `cuda:1` was previously a claim about hardware
nothing enforced.

The obvious implementation — just take `gpu:0` instead of `gpu` — is **wrong in a
way that only shows up in production**. `gpu` and `gpu:0` do not conflict, so an old
run and a new run could land on the same card. Two runs, one GPU, no error, and the
failure surfaces as an OOM that looks like an unrelated bug.

So a run with a known device takes a *pair*:

```python
ResourceRequest("gpu", shared=True)      # "somebody is in the pool"
ResourceRequest(f"gpu:{device}")         # this specific card, exclusively
```

`flock` excludes exclusive against shared, which gives exactly the right
behaviour with no coordination at all:

| in flight | outcome |
|---|---|
| an old run (exclusive `gpu`) | blocks every new run — unchanged from today |
| two new runs, different cards | proceed together |
| two new runs, same card | conflict on the card |
| no device known | falls back to the historical exclusive `gpu` |

**Nothing has to be drained and nothing has to be decided.** While any old run is
in flight the fleet stays serialised exactly as it is now; as old runs finish,
concurrency appears on its own. The conservative outcome is the current behaviour,
so a partially-migrated fleet is never less safe than an un-migrated one.

### The probe asks the same question

`gpu_inventory` takes exactly the lease a run would take, per card. A probe using a
coarser or differently-named lock could report a card as free while the run it is
planning for could not actually have it. A test asserts the two produce identical
requests.

Note the probe had to change too: it previously took the pool lock exclusively,
which under the new scheme would report *every* card busy as soon as one was taken.
Under-reporting availability is safe, but badly wrong.

## Advisory

`PlanAuthority` is all-false with `Literal[False]`, so supplying `True` is a
validation error. Nothing in `gpu_plan` or `gpu_inventory` acquires a lease to
keep — the probe releases before returning. Executing a plan means taking the
`LeaseSet` yourself, which is a decision this module deliberately does not make.
