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

## Proposals require a state change, not a snapshot

`MarketAction` has carried `propose_increase` and `propose_reduce` since the
module was written, and **no code path could return them** — the third dead enum
in this layer, after an `evaluate` stage that `experiment.py` never ran and a
`defer` value with no emitter. A consumer branching on either had dead code, and
the layer could not actually recommend anything.

A snapshot cannot justify a proposal: one print is noise whichever way it moved.
So `InstrumentEvidence` gains `history` — deliberately separate from
`observations`, because two venues quoting at one moment are a *reconciliation*
problem and a series of prints over time is a *state change* problem, and mixing
them would let a two-hour-old print be compared against a fresh one as though both
were current.

`sustained_move` establishes that something changed *and stayed changed*. It
returns nothing unless:

- the series spans real time — two prints seconds apart are one print sampled twice;
- there are enough distinct samples — a venue quoting repeatedly at one instant
  cannot inflate the count into looking like a series;
- there is a net change — up, back down, and flat again has moved nowhere, and
  calling that a change would be inventing one.

This is **not a momentum strategy**. It asks whether the world looks different now
than it did, which is a question about evidence, not a prediction about the future.

### Two bars, because the assets differ

| | bitcoin | index fund |
|---|---|---|
| sustained move required | 8% | 3% |
| venues required | 2 | 1 (consolidated tape) |

The asymmetry follows from the assets rather than from tuning. A one-percent hour in
bitcoin is unremarkable; a sustained three percent in a several-hundred-holding
index fund has no constituent-level explanation, so it is macro news or a bad feed.

### Corroboration counts only venues that are up now

`venue_count` is read from history, which includes feeds that may no longer be
reachable. A proposal whose every supporting feed is dead must not be emitted, so
the attested venues are intersected with the venues that survived the liveness gate.

Losing one of two feeds therefore **removes the proposal but keeps the reading** —
a `propose_hold`, not an `abstain`. Something real did happen and there is a live
venue to act on; only the corroboration is gone. Losing all of them abstains.

A snapshot with no history still never proposes, whatever the price. The
conservative behaviour that predates proposals has not moved.

### A reversal is a state change too

`sustained_move` reports `reversal_up` and `reversal_down` as directions in their
own right. It did not before, and the two available answers were both wrong:

* reporting `None` discards the excursion, so the largest thing that happened in
  the window is invisible;
* reporting the net direction alone calls a failed rally a rising trend, which is
  worse than silence because it is confidently wrong.

`95000 -> 105000 -> 88000` used to read as a plain 7% fall. It is a rally that
failed, and it is now named. Two shapes have to be caught to get this right, and
they need different measures:

* **switched** — the series went *past* its opening price onto the other side
  (rallied before collapsing). Measured from the opening price.
* **failed** — the series traced a direction and handed part of it back from its
  extreme (rose 10%, gave back 6). Measured from the peak or trough.

Measuring only the giveback misses every switch. Measuring only the switch misses
every partial giveback. `excursion_ratio` records how many times over the
counter-move was the size of the move that held, and `net_direction` recovers the
trend that was traced — `reversal_up` is a rise that did not hold, which no
existing field used to say out loud.

A genuinely flat series still reports `None`. The threshold that makes a 10% round
trip legible must not make a 0.1% one legible, or `reversal` stops meaning anything
and every wobbling feed reports a failed move.

**A reversal earns no directional proposal, in any asset.** Reading it as a rise
is the error above. Treating it as the *opposite* direction is worse: that trades
on the assumption a failed rally continues, which is a forecast, and this module
holds no view on what happens next. What it earns is a `propose_hold`.

### History is bounded, because unbounded evidence is evidence about the wrong time

History cannot go through the freshness gate, and that is structural rather than a
shortcut: freshness windows are short by design, so every historical entry would be
rejected as stale and the feature could not exist.

But *unbounded* is not the answer either, and the hazard is specific: an old price
does not merely fail to help, it **inflates**. A series of `50000 from ten days ago`
plus two current prints near 95,000 is a 92% rise on current evidence. Without a
window the detector reports it faithfully and the advisory proposes on it.

So `AssetPolicy.history_window_seconds` bounds how far back a series may reach
(three days by default), `gate_history` enforces it, and the causes are kept
distinct from the quote ones:

| cause | means | operator should |
|---|---|---|
| `history_out_of_window` | the period described has already ended | look at a longer-horizon tool |
| `history_venue_unverified` | the feed behind it cannot be shown reachable | fix the feed |

Conflating these with `stale` would point an operator at the wrong thing: a stale
quote is an observation of *now* that has aged.

History is also not exempt from liveness. Being old does not make a feed
trustworthy — if anything it weakens corroboration, since nothing recent says the
feed is still reporting honestly.

### Provenance, so a proposal can be audited

`MarketDecisionAdvisory.move` carries the signal the decision leaned on. Without it
a proposal cannot be checked afterwards: the reason string says a move was
corroborated but not which move, over what span, from how many prints, or from
which feeds — so two proposals differing only in sample count produce identical
documents.

It is recorded whether or not the move produced a proposal. "A real move happened
and it was not enough" is a finding worth seeing later, and it is invisible if
provenance attaches only to proposals.

## A third asset class: the thin one

`SingleNameEquityPolicy` exists because the first two policies were chosen where
evidence tends to be *good* — bitcoin trades continuously, a broad index fund has a
consolidated tape and a published NAV. Both can mostly be trusted, so each design
question was about freshness machinery.

A thinly traded single name is where that stops being true, and it is where the
framework's stated obligation gets tested:

> `interpret` must abstain rather than guess, and returning a confident proposal
> from thin evidence is the one failure this design exists to prevent.

Every reason it refuses is a reason a naive policy would have spoken:

* **the spread is the evidence.** A move measured mid-spread is mostly the spread
  moving. The bar is therefore expressed *against the spread*, not as a fixed
  percentage — a 5% move in a name quoting 8% wide is not a 5% move. The absolute
  15% floor is a second bar, not the only one.
* **one venue is not corroboration.** A thin name often has exactly one feed, and
  that feed's own series is that feed's story. Reporting it back as a finding is
  reporting the input as the conclusion.
* **there is no index to compare against.** For the index fund a move with no
  constituent-level explanation is news. For a single name *most* moves have a
  company-level explanation this layer knows nothing about, so the same move
  carries far less information.
* **no NAV exists.** The fund publishes a struck price daily; a single name does
  not, so outside the session the last trade is merely the best thing available —
  a weaker statement than the fund's, and the freshness window reflects that.
* **a reversal earns nothing.** A failed rally in a thin name is often one print
  that got hit.

It requires venue liveness, unlike the index fund, because there is no consolidated
tape here — each feed is a counterparty that can stop quoting mid-session, and in a
wide-quoting name that matters because the stale print is also the wide one.

### The fourth dead value, in the file path

`InstrumentBlock` had no `history` field. So `propose_increase` and
`propose_reduce` were unreachable through the CLI two releases after they became
reachable in the core — the same bug as the one that left them unemittable in the
first place, in a different entry point.

That it survived review is the lesson. The core's reachability test asserted
everything about `decide_instrument` and nothing about the document schema, so it
was green for exactly as long as the bug was. A guard that only covers the path it
was written for is not a guard.

The file path now expresses a series, and `history` is a separate field from
`observations` for the same reason it is on `InstrumentEvidence`: two venues
quoting at one moment is reconciliation, a series over time is state change, and a
caller who cannot say which is which cannot be held to either. Every proposal test
in this layer now runs through the file path as well as the core, because a
decision a person is asked to review has to be reproducible from the bytes.

The report also carries `proposed_count`, so a reader need not count actions to
tell "we proposed something" from "we had a reading". It is asserted in both
directions — zero on evidence that only holds, one on evidence that proposes —
because a count that is quietly wrong in the flattering direction is worse than no
count.

### The fifth dead value: an enum nobody set

`PolicyVerdict` was defined, documented at length, exported — and never used. Its
docstring argues for a distinction the code did not make:

> Distinct from an evidence rejection: the quote may be perfectly good and the
> system still unable to say anything, because nobody supplied the rules for this
> instrument. Reporting that as "stale" sends an operator to fix a feed that was
> never broken.

An operator reading an abstention had a prose reason and an evidence-rejection
count, and no way to tell *your feed is broken* from *nobody supplied the rules
for this instrument*. Those are two different repairs.

`MarketDecisionAdvisory.policy_verdict` now carries it, and both reachable members
are emitted:

| verdict | means | operator should |
|---|---|---|
| `no_policy` | no rules were supplied for this instrument | supply a policy |
| `no_basis` | the policy would not name a price field to act on | fix the policy, not the feed |

### The promise in `gate_evidence` that had no "elsewhere"

`gate_evidence` computed a price basis and its own comment said a `None` basis is
"reported as a policy-level abstention **elsewhere** rather than folded in here".
There was no elsewhere. The basis was computed, found to be `None`, and ignored;
the decision then proceeded and produced a document naming **no evidence at all** —
which is the failure the entire provenance design exists to prevent, reached by the
back door. A policy that declines a basis now abstains with `no_basis`.

The first fix only closed that gap in `decide_instrument`. `decide_market_action`, the
single-quote path, still went on to `interpret` with a `None` basis. Both paths now
abstain with `no_basis` and `REASON_NO_BASIS` before `interpret` runs, and each has
its own test.

### Two enum members removed rather than left unused

`no_session` and `policy_abstained` were unreachable by construction, so they are
gone rather than left in place. They were not needed: a session-traded asset
outside its session is not a policy abstention, it is a **hold**, and that is what
the session policies return. Adding an unused hook to justify them would recreate
the exact bug this layer has now found five times. If a future asset genuinely
cannot be reasoned about outside a session, the honest change is to build that path
and its test together.

### Dropped evidence is named in the reason

A discarded series changes what the decision could have been. It used to appear
only in a note, so an operator saw a `propose_hold` whose reason discussed thin
evidence and never learned the series which might have proposed had been thrown
away. It is now in the reason as well as in `rejected`.

### A rejection cause must never be reported as a different one

The primary-reason map existed as **two inline copies**, each falling back to
`REASON_NO_QUOTE` for any cause it did not list. Both history causes added in #26
were unmapped — so had either become the primary reason, an operator would have been
told "no quote supplied for this asset" about a document that supplied one. Nothing
failed. The operator was simply wrong.

Now one `_PRIMARY_REASON`, looked up strictly, with a test asserting it covers every
member. An unmapped cause is a loud `KeyError` in development rather than a wrong
sentence in production.

### Two more unreachable members, and the line they drew

`undated` and `non_positive` were never emitted: `observed_at` is a required
zone-aware datetime and every price field is `gt=0.0`, so neither condition can
arise inside the layer. The type boundary enforces both, which is the stronger
place — an unusable quote never enters rather than being admitted and rejected.
Removed rather than left dead, with the reasoning recorded in the enum.

That draws a line the CLI now states explicitly:

* a **malformed document** is an error the caller must fix;
* a **well-formed quote that fails the gate** is a finding on the advisory.

Conflating them would make a typo in a file look like a decision.

### The file-driven CLI used to answer a bad field with a traceback

`gate_evidence` is not reached by a document containing a non-positive price,
because the model refuses to build it. So a hand-edited file with one bad number
died inside pydantic, and the message said `observations.0.last` — naming the
field but **not which instrument**, in a document that is a list of them.

The refusal now names the instrument, the asset class, and every offending field
rather than the first. And `main` still raises, because a library caller wants the
exception; the new `run` entry point reports it as `error: …` on stderr with exit 2
and no traceback. Both refusals the CLI already made on purpose — unknown schema,
unknown asset class — had the same ugly behaviour and are now clean too.

### A warning on every invocation

A field named `schema` shadowed a deprecated `BaseModel` attribute, so pydantic
emitted a `UserWarning` on *every single run*. A tool that warns every time teaches
its operator to ignore warnings, and this project's position is that the warnings
are the output that matters. Renamed internally with a validation alias; the wire
key is still `schema`, and both facts are asserted.

### Two policies must not hold two copies of a trading calendar

`SingleNameEquityPolicy` shipped with its session boundaries written as
`time(9, 30)` / `time(16, 0)` / `time(4, 0)` / `time(20, 0)` literals while the
index-fund policy imported four named constants for the same four values. They
agreed, so every test passed.

Correct the calendar in one place and the two policies would disagree about whether
the market was open — and this is the layer whose entire purpose is noticing that
two feeds disagree. It now imports the shared constants, and two tests sweep a
trading day in fifteen-minute steps plus a Sunday, and name each boundary
individually so a failure says which one drifted.

### A report that changes bytes between runs is not a report you can audit

`MoveSignal.venues` was a `set[str]`. Set iteration order depends on
`PYTHONHASHSEED`, so the advisory report's bytes differed between two runs over
identical evidence:

```
PYTHONHASHSEED=0   "venues": ["venue-b", "venue-a", "venue-c"]
PYTHONHASHSEED=1   "venues": ["venue-a", "venue-b", "venue-c"]
PYTHONHASHSEED=2   "venues": ["venue-c", "venue-b", "venue-a"]
```

That defeats the entire point of a document input, where the claim is that a
decision can be re-derived later by anyone from the bytes that produced it. Two
reports that differ only in the order of a venue list are the same decision, and a
reviewer cannot tell that without knowing the rule.

The existing reproducibility test could not have caught it: it invoked the CLI
twice *inside one process*, where the hash seed is constant. A test that runs the
same code twice in one process cannot see anything that varies between processes —
so the check is now across processes, which is the only place the difference is
observable at all.

`venues` is an ordered tuple, sorted and duplicate-free by validation rather than
tidied at serialisation. Canonical on the way in means a caller assembling one by
hand fails immediately instead of producing a report whose order depends on
something else.

### Hold is not abstain

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

## Running it

```bash
my-jev-market-decision evidence.json --now 2026-10-06T15:00:00+00:00
```

Input is a document, not a live feed. That is the point: a decision can be
re-derived months later from the exact evidence that produced it, by anyone. A
tool that quietly polls an exchange cannot do that, and a decision whose input has
to be taken on trust is not one anybody can review.

```json
{
  "schema": "my-jev-market-evidence-v1",
  "instruments": [
    {
      "instrument_id": "opaque:btc-spot",
      "asset_class": "crypto_spot_bitcoin",
      "observations": [{"last": 95000.0, "observed_at": "...", "source": "venue-a"}],
      "liveness": [{"venue": "venue-a", "reachable": true, "observed_at": "..."}]
    }
  ]
}
```

Venue liveness is part of the input for the same reason it is part of the gate:
"the quote looked fine" and "there was somebody there to trade it against" are
different facts, and only the first is visible in a price.

Exit code is `0` when at least one instrument was decidable and `2` when every
one abstained, so a caller can tell "we have a reading" from "we have no reading"
without parsing. The abstentions are still printed — they are the useful output
when nothing was decidable.

An unrecognised `asset_class` is **refused**, not abstained on. Quietly abstaining
on a typo would look identical to a deliberate decision not to reason about the
instrument, and only one of those is intended. Same for an unsupported `schema`.

Output is sorted by instrument id, so re-running over the same evidence with the
same `--now` reproduces the same document byte for byte.

## The calendar is partial, and what partial means

`_FIXED_CLOSURES` holds New Year, Independence Day and Christmas, and
`is_fixed_closure` observes them when they land on a weekend: Saturday is observed
the preceding Friday, Sunday the following Monday.

That was the gap this document used to name — *"Independence Day 2026 falls on a
Saturday and the market closes Friday 3 July, which this does not model"* — and it
had a cost worth spelling out, because it was in the expensive direction. On
3 July 2026 the policy reported a **regular** session, so the freshness window was
30 seconds, so a quote carried over from the previous session was judged stale and
the advisory abstained — on a day the market was shut, for a reason an operator
could not check. Spurious abstentions are exactly what teaches people to ignore the
ones that matter.

New Year's Day is why the rule is a function rather than a table. 1 January 2022
fell on a Saturday, so the closure was **Friday 31 December 2021** — a date whose
own nominal holiday is nothing at all. Checking the adjacent day rather than the
adjacent nominal date is what makes that work without re-deriving a table every
December.

The rule is one function with one direction on purpose. The closure test was
duplicated at two call sites, and a rule complicated enough to need explaining is
exactly the kind that gets applied to one and forgotten in the other; the
`_last_weekday_close` site is now asserted directly rather than assumed to follow.

What is still missing is a maintained feed for the movable holidays — Presidents'
Day, Memorial Day, Labor Day, Thanksgiving. Those stay unmodelled and stated rather
than approximated, because a calendar that is confidently wrong produces confidently
wrong session labels. `SingleNameEquityPolicy` shares this calendar rather than
carrying its own copy of the boundaries.

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
### Venue reconciliation, and why it is the same interface twice

Bitcoin has no consolidated tape, so a quote is from one venue and two can
disagree by more than an entire index-fund daily move. `decide_instrument` takes
an `InstrumentEvidence` holding every observation, and `AssetPolicy.reconcile`
gets to choose which to act on.

The two policies use that hook for **opposite** reasons, which is the point of
having it:

| | bitcoin | index fund |
|---|---|---|
| what disagreement means | two prices; one is wrong or illiquid | a broken feed; there is one tape |
| tolerance | 0.5% | 0.1% |
| on breach | abstain, naming **no** preferred venue | abstain, naming it a broken feed |

Preferring the higher bitcoin venue would be picking the feed that suits the
answer, so disagreement declines outright.

**Gating happens before reconciling.** A stale quote from one venue must not be
averaged into a fresh one from another and make a disagreement look like
agreement.

### Corroboration, counted honestly

Independent venues agreeing raises confidence — applied by the core, not by each
policy, because agreement raises confidence for any asset class. Bitcoin moves
from 0.2 with one venue to 0.5 with three.

What does *not* count:

- **the same feed reporting twice** — one voice heard twice
- **two anonymous feeds** — unattributable, so they collapse to one venue
- **one named plus one anonymous** — the anonymous half cannot independently
  confirm anything

Corroboration never pushes confidence above 1.0 and never makes a policy
actionable. It can strengthen a reading; only `interpret` decides whether to act.

When observations are kept, the caveat is kept too. A note saying "repeated
reports from a single feed are not independent agreement" reaches the operator
even though the decision proceeds — dropping it because we proceeded anyway would
lose it entirely.

**The reported price is a venue's own, not a midpoint.** Averaging two venues
would publish a price nobody quoted, which is the same synthetic evidence this
layer refuses elsewhere. The observation nearest the midpoint is chosen, and
`evidence_price` is what that venue actually said. The choice is deterministic:
input order never decides which venue wins.

### Venue liveness: the last gap, closed

A venue that has stopped quoting will serve its last price indefinitely. The
number looks fresh, the quote is well-formed, the spread is plausible, and the
counterparty is gone. For a continuously traded asset that is the characteristic
operational failure, and no amount of price reasoning catches it.

`VenueLiveness` is evidence about a *machine*, gated separately from price:

| state | verdict |
|---|---|
| observed reachable, recent | usable |
| observed unreachable | `venue_down` — rejected outright |
| last seen reachable too long ago | `venue_liveness_stale` — treated as no reading |
| **no liveness evidence at all** | `venue_unverified` — **not** assumed healthy |
| quote names no venue | `venue_unverified` |

Every direction fails closed. "We could not check" and "it is fine" are different
claims, and only the first is supported — so a caller that supplies no liveness
gets an abstention, never a pass. The newest reading wins either way, so a fresh
"down" is not masked by an hour-old "up", and a recovery is seen.

`DEFAULT_LIVENESS_MAX_AGE_SECONDS` is 60. Deliberately much shorter than any
price window: machines fail faster than prices go stale.

`AssetPolicy.requires_venue_liveness` defaults to **True**, so a new asset class
is required to opt out rather than inheriting a silent pass. `IndexFundPolicy`
opts out and says why: a consolidated tape is a subscription, not a counterparty
that vanishes mid-session — the exchange behind it can, and that is the
resolver's problem rather than a quote-evidence one. Bitcoin keeps the default.

Because of these differences bitcoin abstains more readily than the index fund —
confidence ≤ 0.2 against ≤ 0.6 on a single observation — and a test asserts that
ordering. Understating what you know is the cheaper error when the subject is
somebody's savings.

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
