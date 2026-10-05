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

## Adding an asset class

Implement `AssetPolicy`: `freshness_window_seconds`, `session_at`, `price_basis`,
`interpret`. The framework's obligation runs one way — `interpret` must abstain
rather than guess, and returning a confident proposal from thin evidence is the
one failure this design exists to prevent. BTC's policy would return
`session_at = regular` at every hour and a short window, which is the same code
path with a different answer, not a special case.
