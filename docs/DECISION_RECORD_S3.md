# S3 decision record: consolidating the two implementations

_2026-10-04. Supersedes the S3 section of `docs/DECISION_RECORD.md` (PR #19)._

Session 3 was implemented twice, concurrently:

| branch | module | outcome |
|---|---|---|
| `feat/fleet-benchmark-qualification-advisory-20261002` | `fleet_benchmark_qualification.py` | **merged as PR #19** |
| `feat/fleet-benchmark-qualification-advisory-20261002b` | `fleet_qualification.py` | consolidated here, not merged as a module |

An earlier version of this record recommended the opposite direction, on the
reasoning that the `...b` branch was the stronger one. That was wrong: PR #19 had
already merged the `...20261002` branch to main, so the branch being called
"parallel" was the trunk. The recommendation was made against a stale read of
`origin/main`.

## What was kept, and why

The merged implementation stays. It is the trunk, and it is better in two
respects the other branch lacked: `BenchmarkWorkIntent` is a closed enum rather
than an open string map, and health is already a separate input.

Four things were ported onto it rather than reimplemented alongside it.

**Reachability.** `build_fleet_benchmark_advisory` was reachable only from its
own CLI — zero references to "benchmark" existed in `uhp_advisory`,
`heartbeat_compile`, or either CLI. An operator reading a compiled receipt could
not see that the only qualified lane was scout-capable. Now projected as
`SystemOneAdvice.benchmark_qualification`.

**`future_dated` as its own bucket.** Freshness was `0.0 <= age <= max_age`, so
clock skew and expiry both reported as `stale` / "older than the configured TTL".
An operator told "stale" schedules a retest, which cannot help when the
reporter's clock is wrong.

**A handle that is the node id is refused.** `_OPAQUE_HANDLE` matches a bare
hostname, and `_reject_identity` deliberately skips handle values because a real
surrogate embeds the node's slug. So `{"gpu-01.internal.lan":
"gpu-01.internal.lan"}` produced an advisory naming the host. Equality, not
substring, so `eligible:opaque:gpu-01` stays valid.

**A drift-proofed demo.** The committed fixture is pinned in time against a 900s
TTL, so its output was regenerable only in a narrow window and nothing checked.

## What was discarded

`fleet_qualification.py` and `fleet_qualification_cli.py`. Their health
separation was already present and better on main (health is required, not
optional, and gated by a threshold). Their redaction work duplicated what the
port added. Their router bridge consumed a different report shape
(`benchmark-qualification-report-v1`, uppercase roles) than the one main's CLI
takes, so it was never a drop-in.

The `...b` branch is retained only as the tag `pre-rebase-s3-20261004`. It is
local, unpushed, and must not be merged: it writes
`examples/fleet-benchmark-advisory/fleet-health.json` in a schema incompatible
with the one already on main, so a merge would silently break one of them.

## Standing risk

`CONTRACT_SHA256` moves `5e88c73e...` -> `69b9c35d...` because
`SystemOneAdvice` gains a field. Consumers pinning the old digest will see the
new one. This is the one item here with cross-repo blast radius.

Authority, admission, routing, and the deterministic resolver are untouched.
Qualification can withhold implementation and order preference; it cannot grant
dispatch, approval, claims, or mutation.
