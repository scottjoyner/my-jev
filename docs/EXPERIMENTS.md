# Reproducible policy-model experiments

`my-jev` keeps its training lifecycle inside this repository.

The separate `scottjoyner/auto-finetune` project is useful prior art for
resource leases, dataset/model lineage, regression gates and chained ML jobs,
but `my-jev` does not import or depend on it.

## Experiment lifecycle

One TOML spec drives:

```text
dataset manifests
      |
      v
resource leases
      |
      v
training
      |
      v
held-out calibration
      |
      v
test benchmark
      |
      v
promotion gates
      |
      v
run registry + lineage
```

The four data partitions are explicit and immutable within a run:

- **train** — gradient updates;
- **validation** — checkpoint selection;
- **calibration** — temperature fitting only;
- **test** — final benchmark and promotion gate only.

Every split receives a SHA-256 manifest before training starts.

## Encoder baseline

```bash
pip install -e ".[dev]"

my-jev-experiment \
  configs/experiments/assistx-modernbert.toml \
  --dry-run
```

A dry run validates the datasets/spec and writes the exact stage commands
without loading a model.

Run the experiment:

```bash
my-jev-experiment \
  configs/experiments/assistx-modernbert.toml
```

The default encoder experiment uses:

```text
ModernBERT-base
      |
shared state token memory
      |
runtime option queries
      |
typed grouped probabilities
```

## Qwen causal scalar lane

Install the optional causal dependencies:

```bash
pip install -e ".[dev,causal]"
```

Validate the run:

```bash
my-jev-experiment \
  configs/experiments/assistx-qwen35-scalar.toml \
  --dry-run
```

Then train:

```bash
my-jev-experiment \
  configs/experiments/assistx-qwen35-scalar.toml
```

The first causal lane intentionally optimizes for correctness rather than
inference speed:

```text
Qwen3.5-4B-Base
      |
LoRA + scalar sequence-classification head
      |
(state, question, option) cross-encoding
      |
one scalar per option
      |
grouped softmax
```

It does **not** generate answer tokens.

It currently re-encodes the state for every candidate. Shared-state
prefill/cache branching is a subsequent optimization. A cached implementation
must demonstrate numerical/decision equivalence against this reference scorer
before it can replace it.

## Run artifacts

Each run receives a unique directory:

```text
runs/experiments/<run-id>/
├── manifest.json
├── checkpoints/
│   ├── best/
│   └── last/
├── calibration.json
├── benchmark.json
├── promotion.json
└── stages/
    ├── train.command.json
    ├── train.log
    ├── calibrate.command.json
    ├── calibrate.log
    ├── benchmark.command.json
    └── benchmark.log
```

`manifest.json` records:

- experiment spec and its SHA-256;
- train/validation/calibration/test manifests and hashes;
- current git revision/branch/dirty state;
- Python/platform information;
- parent promoted run, when present.

A failed run writes `failure.json` and remains in the registry.

## Resource coordination

The experiment runner uses kernel-backed file leases.

Training requires exclusive leases for:

- `gpu`;
- the specific run output.

Dataset access is shared/read-only.

Registry changes have their own short exclusive lease so multiple agents cannot
silently overwrite lineage metadata.

By default a busy resource fails fast. Use:

```bash
my-jev-experiment SPEC.toml --blocking
```

only when intentionally waiting for the currently active experiment.

## Registry and lineage

The default registry is:

```text
runs/experiments/registry.json
```

Each entry records:

- run ID;
- model backend/backbone;
- dataset hashes;
- checkpoint/calibration/benchmark artifacts;
- promotion status;
- parent promoted run;
- failure notes when applicable.

`parent = "latest_promoted"` in an experiment spec automatically links a new
run to the latest promoted run of that experiment family.

## Promotion is not deployment

A run marked `candidate` passed its experiment gates. It is **not** permission
to route real Hermes actions.

Current gates can include:

- minimum held-out accuracy;
- maximum ECE;
- maximum NLL/Brier;
- minimum accuracy loss when state is shuffled;
- minimum gain over a uniform baseline;
- Choice-order invariance;
- maximum cross-field policy inconsistency;
- latency/throughput bounds.

The initial AssistX specs intentionally omit a throughput gate for the Qwen
cross-encoder because it is a correctness reference. Cache-branching work can
then be judged by whether it improves latency without changing policy quality.

Actual Hermes rollout remains a separate progression:

```text
offline candidate
  -> shadow
  -> advisory
  -> bounded chat/read/task routing
  -> bounded action routing
```

The existing Hermes/AssistX execution and approval gates remain authoritative
at every stage.
