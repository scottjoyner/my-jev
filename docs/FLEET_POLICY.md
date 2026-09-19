# Fleet policy: extending System-One decisions beyond turn routing

The typed policy model can help the fleet without becoming a scheduler or an
authority system. The useful extension is to score **placement shape**: given a
workload and an observed fleet snapshot, estimate which eligible execution
shape is appropriate, while existing Hermes/AssistX controls decide what is
actually allowed and the fleet controller owns leases, claims, starts, stops,
and recovery.

## Separation of responsibilities

```text
request / task
     |
     v
agent policy             learned: what kind of work is this?
     |
     v
fleet placement policy   learned: what execution shape fits?
     |
     v
deterministic resolver   hard constraints + eligibility
     |
     v
fleet controller         authoritative leases / claims / dispatch
     |
     v
node runtime             authoritative execution + health evidence
```

Neither learned layer may:

- mark a node healthy;
- grant a capability;
- override maintenance/drain state;
- acquire or steal a lease;
- bypass an approval;
- dispatch a workload;
- change mutation authority;
- declare recovery complete.

Those facts come from the existing fleet/control plane.

## Candidate typed contract

The first fleet model should stay deliberately small and inspectable.

```text
placement =
  P(local, preferred_node, any_eligible, split, defer, abstain)

resource_shape =
  P(cpu, gpu, memory, io, network, mixed)

latency_class =
  P(interactive, nearline, batch, background)

state_locality =
  P(none, weak, strong, pinned)

migration_tolerance =
  P(free, checkpointable, sticky, immovable)

redundancy =
  P(single, retry_elsewhere, replicated)

fleet_pressure =
  Score(0..4)

placement_confidence =
  Score(0..4)
```

This model predicts **requirements and preferences**, not a hostname. Hostname
selection remains deterministic over the current eligible-node set. That keeps
training labels stable when hardware is added, removed, renamed, drained, or
temporarily unhealthy.

A later experiment can score candidate nodes individually, but only after the
shape model proves useful under shuffled-node and stale-health controls.

## Runtime state supplied to the scorer

Use observable facts only:

- workload type and declared requirements;
- estimated CPU, accelerator, RAM, VRAM, disk, network, and duration;
- interactive/batch deadline;
- checkpoint/restart support;
- data/model locality requirements;
- current eligible-node capabilities;
- measured load and free capacity;
- health freshness;
- drain/maintenance state;
- active workload claims and reservations;
- network reachability class;
- recent failure/retry evidence.

Do not encode hidden reasoning or grant the model access to secrets merely to
improve placement accuracy.

## Deterministic resolver invariants

The resolver intersects learned preferences with authoritative eligibility.

Examples:

- GPU preference + no eligible GPU -> defer or fallback according to the
  workload's declared fallback policy, never invent eligibility.
- preferred node is drained -> remove it before scoring/selection.
- health evidence is stale -> treat the node according to existing fleet
  health policy, not model confidence.
- insufficient free VRAM/RAM -> node is ineligible regardless of score.
- pinned state/data -> locality constraint wins over learned preference.
- existing claim/lease conflict -> controller wins.
- privileged deployment -> existing approval/authority gate wins.

The learned policy may improve ordering among legal choices. It cannot turn an
illegal choice into a legal one.

## Why this helps this fleet

The fleet is heterogeneous. A fixed hostname rule becomes brittle as nodes
change roles and availability. Predicting execution shape gives the controller
a stable semantic layer:

- interactive inference can prefer low-latency available capacity;
- large accelerator jobs can express GPU/VRAM pressure without embedding a
  particular GPU name;
- Neo4j/storage-heavy work can express locality and I/O pressure;
- long recovery or benchmark jobs can be marked sticky/checkpointable;
- low-priority generation/ingestion can yield under fleet pressure;
- workloads can be retried elsewhere only when their semantics permit it.

The controller can then map those requirements onto whatever the fleet looks
like at that moment.

## Evidence loop

Start in observer mode beside the current scheduler.

For every placement decision capture:

```text
workload state
fleet snapshot + freshness
eligible nodes after hard constraints
current controller placement
candidate typed distribution
candidate resolved preference
actual node
queue/start/finish timestamps
resource peaks
failure/retry/preemption
verification outcome
operator override/correction
```

Candidate output is evidence only.

Useful labels come from verified outcomes rather than copying today's routing
rules. Examples include successful completion, deadline met/missed, OOM,
accelerator fallback, unnecessary transfer, retry success, operator relocation,
and measured queue/runtime cost.

## Evaluation controls

A fleet scorer needs stronger controls than ordinary classification accuracy.

1. **Node permutation** — reorder eligible nodes; shape predictions must remain
   stable.
2. **Hostname ablation** — remove hostnames entirely. The model should learn
   capabilities and state, not memorized machines.
3. **Shuffled fleet state** — swap utilization/health snapshots; decisions that
   depend on pressure should degrade.
4. **Stale-health counterfactual** — same node metrics, different freshness;
   deterministic eligibility must change correctly.
5. **Drain counterfactual** — same workload/node, drained vs available; drained
   nodes must never survive the resolver.
6. **Capacity boundary** — perturb RAM/VRAM around required thresholds and prove
   hard constraints dominate model preference.
7. **New-node holdout** — evaluate on hardware identities absent from training.
8. **Failure replay** — replay OOM, unreachable, lease-conflict, and node-loss
   trajectories and verify safe fallback/defer behavior.

## Promotion ladder

Keep promotion incremental:

```text
offline historical replay
        ->
live shadow scoring
        ->
operator-visible recommendations
        ->
bounded ordering among already-eligible nodes
        ->
optional controller integration
```

Do not jump directly from benchmark accuracy to autonomous fleet dispatch.

The first integration milestone should only let the model reorder nodes already
declared eligible by authoritative fleet logic. Existing controller behavior
remains the fallback whenever confidence/evidence is insufficient.

## Implemented observer-mode slice

The second typed dataset family now exists as `fleet-placement-v1`, separate
from the original agent-policy targets.

Implemented pieces:

- typed fleet placement contract with anonymous node facts;
- deterministic hard eligibility for health, freshness, drain, reachability,
  capability, RAM/VRAM capacity, and pinned locality;
- deterministic observer-only resolver that can rank only already-eligible
  nodes and always returns `dispatch_allowed=false`;
- low-confidence abstention plus safe defer behavior for impossible local/split
  recommendations;
- verifier-labeled counterfactual families covering node permutation,
  preferred-node drain, stale pinned health, VRAM capacity boundaries, and
  checkpoint support;
- provenance isolation: `family_id`, label source, scenario, and variant never
  enter the model state;
- one-command group-safe train/validation/calibration/test preparation through
  `my-jev-fleet-prepare`;
- independent ModernBERT experiment spec at
  `configs/experiments/fleet-modernbert.toml`.

Example:

```bash
my-jev-fleet-prepare \
  --output-dir data/fleet-placement-v1 \
  --records 12000 \
  --seed 31

my-jev-experiment \
  configs/experiments/fleet-modernbert.toml \
  --dry-run
```

Keeping the task family separate gives fleet placement independent calibration
and promotion gates. A later multi-task checkpoint can share the encoder only
if evidence shows that it improves both policies rather than causing
interference.
