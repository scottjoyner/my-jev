# R9700 ModernBERT baseline

This is the smallest checked-in **full-encoder** baseline intended to establish
the first exact-code-SHA ModernBERT checkpoint on the AMD Radeon AI PRO R9700.

It is an offline training/evaluation run only. It does not change Hermes or
AssistX execution authority, approval gates, speaker verification, mutation
permissions, tool guardrails, claim/fencing, or recovery controls.

## Why this slice exists

The normal ModernBERT experiment is sized for the real bootstrap corpus. Before
spending a longer run on it, this slice proves the entire artifact path on the
R9700:

```text
exact clean git SHA
    -> HIP/BF16/R9700 preflight
    -> deterministic group-safe dataset
    -> full ModernBERT encoder fine-tune
    -> best checkpoint
    -> held-out calibration
    -> held-out benchmark
    -> promotion evaluation
    -> artifact receipt with SHA-256s
```

The model may be rejected by the normal quality gates and the run is still
useful. The milestone is an exact-SHA checkpoint plus calibration and benchmark
artifacts, not permission to deploy or route actions.

## Pinned small-run contract

The checked-in spec is:

```text
configs/experiments/assistx-modernbert-r9700-baseline.toml
```

It keeps the production-shape encoder settings while reducing only the amount
of training work:

- backbone: `answerdotai/ModernBERT-base`
- option-query head rank: 256
- source records: 2,048
- generator seed: 23
- train seed: 17
- epochs: 1
- batch size: 2
- gradient accumulation: 8
- state length: 2,048
- candidate length: 192
- BF16: required
- backbone: trainable
- gradient checkpointing: enabled
- benchmark/calibration batch size: 4

The dataset is generated under `runs/datasets/`, which is already ignored by
git. That lets the experiment manifest truthfully record `dirty=false`.

## Exact-SHA launch

Start from the exact PR commit you intend to measure. Do not use an implicit
moving branch head as the identity of the experiment.

```bash
git fetch origin
git checkout <exact-40-character-sha>

python -m pip install -e ".[dev]"

bash scripts/run-r9700-modernbert-baseline.sh \
  <exact-40-character-sha>
```

The launcher refuses to continue when:

- `HEAD` is not the supplied SHA;
- the checkout is dirty;
- Python imports `my_jev` from anywhere other than this checkout;
- PyTorch cannot see an accelerator;
- BF16 is unavailable;
- PyTorch does not report a HIP runtime;
- no visible device name contains `R9700`;
- the matching device reports less than 28 GiB;
- a pre-existing deterministic dataset has a different generator/seed/count;
- the experiment fails to create the required artifact set;
- the final checkpoint reports CPU/non-BF16/frozen-backbone/wrong run settings;
- the experiment manifest does not record the supplied SHA and a clean tree.

If the ROCm device string on a particular host does not literally contain
`R9700`, override only the device-name match while preserving the memory and
HIP checks:

```bash
MY_JEV_R9700_DEVICE_PATTERN="<actual rocM device substring>" \
  bash scripts/run-r9700-modernbert-baseline.sh <exact-sha>
```

The minimum visible memory check defaults to 28 GiB and can be changed with
`MY_JEV_R9700_MIN_GIB` for diagnostics. A changed value should be recorded
when interpreting the result.

## Artifact acceptance

A successful launcher ends with a run directory under:

```text
runs/experiments/assistx-policy-modernbert-r9700-baseline-*/
```

Required artifacts are:

```text
manifest.json
checkpoints/best/model.pt
checkpoints/best/my_jev_config.json
calibration.json
benchmark.json
promotion.json
r9700-doctor.json
pip-freeze.txt
r9700-baseline-receipt.json
stages/train.log
stages/calibrate.log
stages/benchmark.log
```

`r9700-baseline-receipt.json` records the exact git SHA, experiment-spec hash,
all four dataset hashes, and SHA-256/byte size for the core output artifacts.

The checkpoint config is also re-read after training to prove that the measured
run actually used CUDA/HIP through PyTorch, BF16, a trainable encoder, one epoch,
batch size 2, accumulation 8, the 2,048-token state window, and gradient
checkpointing.

## What comes next

Do not widen runtime authority after this run.

Use the resulting benchmark first to answer:

1. Does the small baseline beat uniform behavior and materially degrade under
   shuffled state?
2. Are calibration and Choice-order invariance sane enough to justify the
   50,000-record run?
3. Are cross-field policy-consistency violations within the existing gate?
4. What are R9700 states/sec, decisions/sec, and p95 batch latency?
5. Does the checkpoint produce useful disagreements when replayed in
   Hermes/AssistX shadow mode?

Only after those offline/shadow checks should a larger training run be queued.
Runtime execution remains governed by the existing deterministic resolver and
Hermes/AssistX controls.
