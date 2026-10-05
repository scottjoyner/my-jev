# Market decision advisory

Answers one narrow question about market evidence: **does what I have in hand
justify proposing anything at all?** It does not forecast prices and does not
place orders.

```python
from my_jev.market_decision import MarketQuote, decide_market_action
from my_jev.market_asset_index_fund import IndexFundPolicy

advisory = decide_market_action(quote, IndexFundPolicy(), now=when)
```

`advisory.action` is a proposal. Placing an order is an external side effect, so
`TradingAuthority` is all-false with `Literal[False]` — supplying `True` anywhere
on it is a validation error, not an ignored field.

## Why freshness is asset-class aware

This is the design decision everything else follows from. A single
"evidence older than N minutes is stale" rule is **correct** for a continuously
traded instrument and **wrong** for an exchange-traded fund:

- **BTC** trades 24/7. Any wall-clock instant is a plausible observation, so
  freshness genuinely is elapsed time.
- **A US index fund** trades 09:30–16:00 America/New_York. Outside those hours
  the most recent trade is from the *previous session* — valid evidence, simply
  from a different session than the one you are asking about.

A naive elapsed-time rule marks every overnight quote stale. The system then
abstains constantly and correctly, which teaches an operator to ignore it. That
is how a correct abstention becomes a useless one.

So `AssetPolicy.freshness_window_seconds(now)` takes the clock, not just a
duration:

| moment (2026-10-06) | session | window |
|---|---|---|
| 11:00 ET | `regular` | 300 s |
| 04:30 ET | `pre_market` | 2,100 s |
| 17:00 ET | `after_hours` | 3,900 s |
| Sat 12:00 ET | `closed` | 72,300 s |
| Mon 03:00 ET (overnight) | `closed` | 39,900 s |

The last two are the point. The same quote that is correctly accepted overnight
becomes correctly rejected once a new session opens:

```text
last close, seen Mon 22:00 ET  ->  propose_hold, rejected {}      (valid evidence)
last close, seen Tue 12:00 ET  ->  abstain,      rejected stale  (session moved on)
```

## Rejections name their cause

`EvidenceRejection` carries one value per fault, and every cause is collected
rather than short-circuiting on the first — an operator fixing a feed wants the
whole list:

`absent` · `undated` · `future_dated` · `stale` · `non_positive` · `crossed` ·
`ambiguous_price`

`future_dated` is deliberately separate from `stale`. A quote from the future is
clock skew on the reporting host, and the fix is not "wait for fresher data".

A **missing policy** is also not an evidence rejection. With no policy there is no
freshness rule to apply, so reporting the quote as `stale` would send an operator
to fix a feed that was never broken.

## The index-fund policy, and what it refuses

`IndexFundPolicy` models a broad US equity index fund. It is the first asset class
because its failure mode is the least dramatic and the most reliably measured:
the harm that actually reaches people holding a broad index fund is selling out
during a drawdown, not a missed rally.

- **NAV is not a last trade.** The fund publishes an official NAV struck on the
  underlying basket's closing prices. During the session the trade price is what
  an execution would meet; outside it, the NAV. The chosen field is reported in
  `price_basis` and the value in `evidence_price`.
- **The calendar, not the feed, decides the session.** A quote that declares
  itself `regular` at 07:00 ET does not get the intraday basis. The declaration
  is the untrusted input.
- **No constituent-level catalyst.** A large move in one holding of several
  hundred is not a reason to change a position in the fund containing it. The
  policy structurally cannot propose a trade on one.
- **A trade price far from the NAV is a broken feed**, not a move. Acting on
  either would be acting on bad data.

The bar is asymmetric on purpose: ordinary volatility yields `propose_hold`,
never a proposal. `intraday_move_threshold` is a constructor field rather than a
constant so a caller can tighten it per instrument, visibly.

## Hold is not abstain

```
MarketAction.abstain           not enough to say anything
MarketAction.propose_hold      evidence says: no change
MarketAction.propose_increase  proposal, requires human approval
MarketAction.propose_reduce    proposal, requires human approval
```

Collapsing "nothing to do" and "not enough to say" into one value is how a system
that rarely speaks up ends up trusted when it does. `advisory.actionable` is true
only for the two proposals.

## It records its evidence

`evidence_price`, `evidence_observed_at` and `evidence_source` are on the
advisory. Without them two decisions made from different quotes produce identical
documents and neither can be audited after the fact.

`price_forecast` is permanently `False`. It exists so a reader cannot mistake
this for a signal. `decision_confidence` is confidence in the recommendation to
propose, never in a direction.

## The calendar is partial, deliberately

`_FIXED_CLOSURES` holds New Year, Independence Day and Christmas. A real
deployment needs a maintained calendar feed for observed holidays — for example
Independence Day 2026 falls on a Saturday and the market closes Friday 3 July,
which this does not model. That is stated rather than approximated, because a
calendar that is confidently wrong produces confidently wrong session labels.

## Bitcoin: the same core, a different answer

`BitcoinSpotPolicy` was written to *test* the extension point rather than to
broaden it. If a 24/7, no-NAV, order-of-magnitude-more-volatile asset needed its
own decision function, `AssetPolicy` would be decoration.

It didn't. One core decides for both, and a test asserts the core contains no
reference to either asset class, so the next person cannot quietly special-case
one.

What actually differs, and it is mostly *simpler*:

| | index fund | bitcoin |
|---|---|---|
| calendar | Eastern, sessions, partial holidays | none at all |
| session | `regular` / `pre_market` / `after_hours` / `closed` | always `regular` |
| freshness | 300 s in session, opening up overnight | always 30 s |
| reference price | official NAV | none exists |
| move threshold | 1% | 10% |

No calendar means no timezone, no session boundaries, and no holiday list that is
wrong about observed holidays. All of that complexity is absent because the
domain lacks it. And the freshness window stays tight precisely because a current
price always exists — the widening in the index-fund policy exists only because
overnight no newer observation *can* exist.

Thin weekend liquidity is deliberately **not** folded into freshness. Widening the
window to excuse an old quote would blur "stale" into "quiet", and an operator
needs to know which they were told.

### Where the abstraction genuinely strains

Two gaps, recorded rather than worked around:

**There is no consolidated tape.** A bitcoin quote is from one venue and two
venues can disagree by more than an entire index-fund daily move. `AssetPolicy`
evaluates one `MarketQuote` and never sees a second, so it cannot compare venues
or notice a dislocation. An index fund does not need this: it has one tape and an
official NAV. Fixing it properly means widening the interface to accept several
observations per instrument — a real change, not smuggled in here.

**Custody and venue liveness are not prices.** An exchange going down during
volatility is the characteristic operational failure and nothing here sees an
exchange's status. That is venue liveness, a different kind of check, and
pretending a price model covers it would be worse than omitting it.

Because of both, bitcoin abstains more readily than the index fund — confidence
≤ 0.2 against ≤ 0.6 — and a test asserts that ordering. Understating what you know
is the cheaper error when the subject is somebody's savings.

### The thresholds are descriptive, not gating

`normal_move_threshold` and `thin_liquidity_move_threshold` appear in the
operator-facing reason so the rule that applied is visible, but they do not
currently change the action: one print is never sufficient regardless of how far
it moved. Mutation-testing confirmed narrowing the threshold to the index-fund
scale passes every behavioural test. That is pinned by a test which asserts both
halves — the threshold is reported, and the action is unchanged — rather than
left to be rediscovered.

## Adding an asset class

Implement `AssetPolicy`: `freshness_window_seconds`, `session_at`, `price_basis`,
`interpret`. The framework's obligation runs one way — `interpret` must abstain
rather than guess, and returning a confident proposal from thin evidence is the
one failure this design exists to prevent. BTC's policy would return
`session_at = regular` at every hour and a short window, which is the same code
path with a different answer, not a special case.
