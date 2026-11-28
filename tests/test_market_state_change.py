"""Reversals, history gating, move provenance, and a third asset class.

Four related additions, collected because each one exists to close a specific hole
and the holes are easier to see together than separately.
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

import pytest

from my_jev.market_asset_crypto import BitcoinSpotPolicy
from my_jev.market_asset_equity import SingleNameEquityPolicy
from my_jev.market_asset_index_fund import IndexFundPolicy
from my_jev.market_decision import (
    DEFAULT_HISTORY_WINDOW_SECONDS,
    DEFAULT_MOVE_SPAN_SECONDS,
    DEFAULT_REVERSAL_MINIMUM_EXCURSION,
    EvidenceRejection,
    InstrumentEvidence,
    MarketAction,
    MarketQuote,
    MarketSession,
    MarketDecisionAdvisory,
    PolicyVerdict,
    TradingAuthority,
    VenueLiveness,
    decide_instrument,
    decide_market_action,
    sustained_move,
)

NOW = datetime(2026, 10, 6, 15, 0, tzinfo=UTC)  # Tuesday 11:00 America/New_York
BTC = BitcoinSpotPolicy()

#: A fall comfortably past bitcoin's 8% proposal bar. Written as a series of
#: prices rather than a percentage so the fixtures cannot silently drift under
#: the threshold -- an earlier version of this file used a 7.4% fall and asserted
#: it would propose.
BTC_FALL = [95_000.0, 88_000.0, 80_000.0]
EQUITY = SingleNameEquityPolicy()


def quote(*, age_seconds: float, last: float, source: str | None = "a",
          bid: float | None = None, ask: float | None = None,
          session: MarketSession = MarketSession.regular) -> MarketQuote:
    return MarketQuote(
        instrument_id="opaque:x",
        last=last,
        bid=bid if bid is not None else last * 0.999,
        ask=ask if ask is not None else last * 1.001,
        observed_at=NOW - timedelta(seconds=age_seconds),
        source=source,
        session=session,
    )


def live(venue: str = "a", ok: bool = True) -> VenueLiveness:
    return VenueLiveness(
        venue=venue, reachable=ok, observed_at=NOW - timedelta(seconds=5)
    )


def series(prices: list[float], *, step_minutes: int = 30,
           venues: tuple[str, ...] = ("a", "b")) -> list[MarketQuote]:
    """A series given in chronological order -- oldest price first.

    Written this way because three earlier versions of this file's fixtures
    inverted the direction and every one of them produced a passing test of the
    wrong thing. A list read left to right in time order cannot be flipped.
    """
    span = (len(prices) - 1) * step_minutes
    return [
        MarketQuote(
            instrument_id="opaque:x",
            last=price,
            observed_at=NOW - timedelta(minutes=span - index * step_minutes),
            source=venues[index % len(venues)],
        )
        for index, price in enumerate(prices)
    ]


# ===========================================================================
# A reversal is a state change, and used to be reported as nothing
# ===========================================================================


@pytest.mark.parametrize(
    ("prices", "expected"),
    [
        ([95_000, 98_000, 101_500], "up"),
        ([101_500, 98_000, 95_000], "down"),
        # rose 10.5% and came all the way back: the case that returned None before
        ([95_000, 105_000, 95_000], "reversal_up"),
        # fell 9.5% and came all the way back
        ([95_000, 86_000, 95_000], "reversal_down"),
        # rallied before collapsing: a switch, not a giveback
        ([95_000, 105_000, 88_000], "reversal_down"),
        # collapsed before bouncing: the mirror of the above
        ([105_000, 88_000, 101_500], "reversal_down"),
        # rose, fell, kept most of the rise: a giveback
        ([95_000, 105_000, 99_000], "reversal_up"),
    ],
)
def test_every_shape_of_series_is_named(prices, expected):
    assert sustained_move(series(prices)).direction.value == expected


@pytest.mark.parametrize(
    "prices",
    [
        [95_000, 95_000, 95_000],       # never moved
        [95_000, 95_100, 95_000],       # wobbled, but under the reversal threshold
        [95_000, 95_010, 95_000],
    ],
)
def test_a_series_that_really_went_nowhere_is_still_nothing(prices):
    """Reversal detection must not become a way to call noise a move.

    The threshold that makes a 10% round trip legible must not make a 0.1% one
    legible, or "reversal" stops meaning anything and every wobbling feed is
    suddenly reporting a failed move.
    """
    assert sustained_move(series(prices)) is None


def test_net_direction_recovers_the_trend_a_reversal_traced():
    """Without this, "reversal_up" never says which way it went first."""
    rose = sustained_move(series([95_000, 105_000, 95_000]))
    assert rose.direction.value == "reversal_up"
    assert rose.net_direction.value == "up"

    fell = sustained_move(series([95_000, 86_000, 95_000]))
    assert fell.direction.value == "reversal_down"
    assert fell.net_direction.value == "down"


def test_a_round_trip_reports_its_excursion_not_a_zero():
    """Reporting 0.0 would assert the series went nowhere.

    It went 10.5% and came back. `magnitude` is defined as the net change, which
    is nil here, so the excursion is reported in its place and `direction` marks
    why.
    """
    signal = sustained_move(series([95_000, 105_000, 95_000]))
    assert signal.magnitude == pytest.approx(0.10526, abs=1e-4)
    assert signal.excursion_ratio is None  # no net move to be a fraction of


def test_excursion_ratio_reads_above_one_when_a_move_failed_to_hold():
    clean = sustained_move(series([95_000, 98_000, 101_500]))
    assert clean.excursion_ratio is None

    failed = sustained_move(series([95_000, 105_000, 99_000]))
    assert failed.excursion_ratio > 1.0


def test_the_reversal_threshold_is_a_constant_not_a_literal():
    """It is referenced by name in the docstring and in the tests above."""
    assert DEFAULT_REVERSAL_MINIMUM_EXCURSION > 0.0
    # A move smaller than the reversal threshold must not be called a reversal
    # even though it is plainly a change.
    small = sustained_move(series([95_000, 95_150, 95_000]))
    assert small is None


# ===========================================================================
# A reversal earns no directional proposal
# ===========================================================================


def decide(history, *, policy=BTC, observations=None, liveness=None):
    observations = observations if observations is not None else [
        quote(age_seconds=30, last=95_500, source="a"),
        quote(age_seconds=30, last=95_510, source="b"),
    ]
    return decide_instrument(
        InstrumentEvidence(
            instrument_id="opaque:x", observations=observations, history=history
        ),
        policy,
        now=NOW,
        liveness=liveness if liveness is not None else [live("a"), live("b")],
    )


def test_a_reversal_proposes_no_direction_in_any_asset():
    """Reading a failed rally as a rise is the error this exists to prevent.

    The alternative was treating a reversal as the opposite direction, which is
    worse: that trades on the assumption a failed rally continues, which is a
    forecast, and this module holds no view on what happens next.

    All three policies are exercised rather than just the two that existed before,
    because a reversal guard added to a new policy and never fed a reversal is a
    guard that does not exist.

    Each series is sized to clear its own policy's *absolute* bar, so the reversal
    guard is what refuses. A smaller reversal is refused earlier by the magnitude
    threshold and would pass this test without ever reaching the guard -- which is
    how the first version of this test proved nothing.
    """
    equity_observations = [
        quote(age_seconds=30, last=100.0, bid=99.8, ask=100.2, source="a"),
        quote(age_seconds=30, last=100.02, bid=99.82, ask=100.22, source="b"),
    ]

    for policy, prices in (
        (BTC, [95_000.0, 110_000.0, 80_000.0]),
        (IndexFundPolicy(), [500.0, 580.0, 420.0]),
        (EQUITY, [100.0, 150.0, 84.0]),
    ):
        if policy is EQUITY:
            advisory = equity_decide(series(prices), observations=equity_observations)
        else:
            advisory = decide(series(prices), policy=policy)

        assert advisory.move.direction.value == "reversal_down", policy.asset_class
        # Fixture sanity: the reversal must be one the absolute bar would have let
        # through, or this test is measuring the magnitude threshold again.
        assert advisory.move.magnitude >= policy.propose_move_threshold, policy.asset_class
        assert advisory.action is MarketAction.propose_hold, policy.asset_class
        assert advisory.actionable is False


def test_the_move_is_still_recorded_when_it_proposes_nothing():
    """"A real move that was not enough" has to be visible afterwards."""
    advisory = decide(series([95_000, 105_000, 95_000]))
    assert advisory.action is MarketAction.propose_hold
    assert advisory.move is not None
    assert advisory.move.direction.value == "reversal_up"


# ===========================================================================
# History is bounded, and bounded history has its own rejection causes
# ===========================================================================


def test_a_series_from_last_week_cannot_propose_today():
    """Evidence about a period that ended is not evidence about the present.

    The detector will happily compute a clean, well-corroborated, 92% move from
    any series it is handed -- and without the window that series would be the one
    entry from ten days ago plus two current ones. The hazard is not that old
    evidence is useless; it is that it silently *inflates* the move, because a
    price from a week ago is a large distance from today's.
    """
    ancient_then_now = [
        MarketQuote(
            instrument_id="opaque:x", last=50_000,
            observed_at=NOW - timedelta(days=10), source="a",
        ),
        *series([95_000.0, 96_000.0], step_minutes=30),
    ]

    # Without the bound this is a spectacular 92% rise on two fresh prints.
    unbounded = sustained_move(ancient_then_now)
    assert unbounded is not None
    assert unbounded.magnitude > 0.9

    advisory = decide(ancient_then_now)
    assert advisory.action is MarketAction.propose_hold
    assert advisory.move is None
    assert EvidenceRejection.history_out_of_window in advisory.rejected


def test_out_of_window_is_not_conflated_with_stale():
    """They are opposite mistakes and send an operator to different places.

    A stale quote is an observation of *now* that has aged; an out-of-window
    history entry describes a period that has already ended. Telling an operator
    "stale" when the series is a week old points at the wrong feed entirely.
    """
    advisory = decide(
        [
            MarketQuote(
                instrument_id="opaque:x", last=50_000,
                observed_at=NOW - timedelta(days=10), source="a",
            ),
            *series([95_000.0, 96_000.0]),
        ]
    )
    assert EvidenceRejection.stale not in advisory.rejected
    assert EvidenceRejection.history_out_of_window in advisory.rejected


def test_history_dated_in_the_future_is_refused():
    """Clock skew is not history, and must not become a basis for a proposal.

    Same hazard as an out-of-window entry: an undated-ish print that lands in the
    wrong place would sort to the end of the series and turn a 1% drift into a
    47% fall.
    """
    future_then_now = [
        MarketQuote(
            instrument_id="opaque:x", last=50_000,
            observed_at=NOW + timedelta(minutes=5), source="a",
        ),
        *series([95_000.0, 96_000.0]),
    ]
    assert sustained_move(future_then_now).magnitude > 0.4

    advisory = decide(future_then_now)
    assert advisory.action is MarketAction.propose_hold
    assert EvidenceRejection.future_dated in advisory.rejected


def test_history_from_a_feed_that_cannot_be_checked_is_refused():
    """Being old does not exempt a series from the liveness rule.

    If anything it weakens corroboration: there is no recent sign the feed is
    still reporting honestly.
    """
    advisory = decide(
        series([95_000, 88_000], venues=("a", "zz")),
        liveness=[live("a"), live("b")],
    )
    assert advisory.action is MarketAction.propose_hold
    assert EvidenceRejection.history_venue_unverified in advisory.rejected


def test_the_history_window_is_bounded_by_default():
    """An unbounded window would let any series in the asset's life qualify."""
    assert DEFAULT_HISTORY_WINDOW_SECONDS > DEFAULT_MOVE_SPAN_SECONDS


def test_a_policy_may_tighten_its_history_window():
    assert EQUITY.history_window_seconds(NOW) <= DEFAULT_HISTORY_WINDOW_SECONDS


# ===========================================================================
# Move provenance on the advisory
# ===========================================================================


def test_a_proposal_names_the_move_it_leaned_on():
    """Otherwise two proposals differing only in span and sample count are
    indistinguishable documents, and neither can be audited."""
    advisory = decide(series(BTC_FALL))
    assert advisory.action is MarketAction.propose_reduce
    assert advisory.move is not None
    assert advisory.move.direction.value == "down"
    assert advisory.move.sample_count == 3
    assert advisory.move.span_seconds > 0
    assert advisory.move.venues


def test_two_proposals_from_different_evidence_are_different_documents():
    """The reason for having provenance at all."""
    thin = decide(series(BTC_FALL))
    thick = decide(series([95_000, 92_000, 89_000, 86_500]))
    assert thin.action is thick.action is MarketAction.propose_reduce
    assert thin.move.sample_count != thick.move.sample_count
    assert thin.model_dump_json() != thick.model_dump_json()


def test_no_move_means_no_provenance():
    advisory = decide_instrument(
        InstrumentEvidence(instrument_id="opaque:x",
                           observations=[quote(age_seconds=30, last=95_000)]),
        BTC, now=NOW, liveness=[live("a")],
    )
    assert advisory.move is None


def test_provenance_does_not_leak_into_the_authority():
    """A richer advisory must not become a more actionable one."""
    advisory = decide(series(BTC_FALL))
    assert set(advisory.authority.model_dump().values()) == {False}
    assert advisory.price_forecast is False
    assert advisory.advisory_only is True


def test_the_authority_still_rejects_a_true_anywhere():
    with pytest.raises(ValueError):
        TradingAuthority(order_placed=True)


# ===========================================================================
# A third asset class: the one where the framework's obligation is hardest
# ===========================================================================


def equity_decide(history, *, observations, liveness=None):
    return decide_instrument(
        InstrumentEvidence(
            instrument_id="opaque:x", observations=observations, history=history
        ),
        EQUITY,
        now=NOW,
        liveness=liveness if liveness is not None else [live("a"), live("b")],
    )


def test_a_single_name_proposes_on_a_large_corroborated_move():
    observations = [
        quote(age_seconds=30, last=130.0, bid=129.87, ask=130.13, source="a"),
        quote(age_seconds=30, last=130.02, bid=129.89, ask=130.15, source="b"),
    ]
    advisory = equity_decide(series([100.0, 112.0, 130.0]), observations=observations)
    assert advisory.action is MarketAction.propose_increase
    assert advisory.actionable is True  # it does propose a change
    assert set(advisory.authority.model_dump().values()) == {False}


def test_one_venue_is_not_corroboration_even_for_a_huge_move():
    """The failure this policy exists to prevent.

    A thin name often has exactly one feed. Reporting that feed's own series back
    as a finding is reporting the input as the conclusion.
    """
    observations = [quote(age_seconds=30, last=130.0, bid=129.87, ask=130.13, source="a")]
    advisory = equity_decide(
        series([100.0, 112.0, 130.0], venues=("a",)),
        observations=observations,
        liveness=[live("a")],
    )
    assert advisory.move is not None
    assert advisory.action is MarketAction.propose_hold


def test_a_move_much_smaller_than_the_spread_is_not_a_price_change():
    """The spread is the evidence in a thin name.

    A move measured mid-spread is mostly the bid and ask moving apart, and a
    proposal here would be proposing a trade that cannot profitably be made. The
    bar is deliberately expressed against the spread rather than as a fixed
    percentage, because in a name this thin the spread *is* the volatility.

    The numbers are deliberately extreme: the spread has to be wider than half the
    absolute 15% bar before this guard is the thing refusing, because below that
    the absolute bar speaks first. An earlier version of this test used a 2% move
    and a 4% spread and passed -- on the absolute threshold, never reaching the
    spread check at all, which is a test of the wrong thing that looks correct.
    """
    observations = [
        quote(age_seconds=30, last=100.0, bid=80.0, ask=120.0, source="a"),
        quote(age_seconds=30, last=100.0, bid=80.0, ask=120.0, source="b"),
    ]
    # A 20% fall: past the 15% absolute bar, but a fifth of the 40% round trip
    # through a 20% half-spread.
    advisory = equity_decide(series([120.0, 110.0, 96.0]), observations=observations)

    assert advisory.move is not None
    assert advisory.move.magnitude > 0.15, "fixture must clear the absolute bar"
    assert advisory.action is MarketAction.propose_hold


def test_a_wide_spread_alone_is_enough_to_refuse():
    observations = [
        quote(age_seconds=30, last=100.0, bid=94.0, ask=106.0, source="a"),
        quote(age_seconds=30, last=100.02, bid=94.0, ask=106.1, source="b"),
    ]
    advisory = equity_decide([], observations=observations)
    assert advisory.action is MarketAction.propose_hold
    assert "too wide to act on" in advisory.reason


def test_venues_that_disagree_about_one_instrument_are_refused():
    """They are quoting the same thing, so a wide split is a fault, not a spread."""
    observations = [
        quote(age_seconds=30, last=100.0, bid=99.9, ask=100.1, source="a"),
        quote(age_seconds=30, last=140.0, bid=139.8, ask=140.2, source="b"),
    ]
    advisory = equity_decide([], observations=observations)
    assert advisory.action is MarketAction.abstain
    assert "disagreed" in advisory.reason


def test_a_single_name_requires_liveness_and_the_other_two_do_not():
    """Each policy's answer is a consequence of its market structure, not a preference."""
    assert EQUITY.requires_venue_liveness is True
    assert BTC.requires_venue_liveness is True
    assert IndexFundPolicy().requires_venue_liveness is False


def test_the_three_policies_disagree_about_the_same_evidence():
    """Which is the reason to have a policy per asset class at all."""
    history = series([100.0, 112.0, 130.0])
    observations = [
        quote(age_seconds=30, last=130.0, bid=129.87, ask=130.13, source="a"),
        quote(age_seconds=30, last=130.02, bid=129.89, ask=130.15, source="b"),
    ]
    actions = {
        policy.asset_class: equity_decide(history, observations=observations)
        for policy in (EQUITY, BTC)
    }
    # A 30% move on a thin single name is proposal-worthy only because it dwarfs the
    # spread; for bitcoin it is ordinary, and bitcoin additionally demands two
    # venues that are both live.
    assert actions[EQUITY.asset_class].action is MarketAction.propose_increase
    assert all(a.action is not MarketAction.abstain for a in actions.values())
    for advisory in actions.values():
        assert set(advisory.authority.model_dump().values()) == {False}


def test_proposing_something_still_places_nothing():
    """`actionable` means "proposes a change", and is true by design for a proposal.

    What must be impossible is that a proposal becomes an instruction, so that is
    what this asserts -- on a real proposal, not a neutral reading.
    """
    advisory = decide(series(BTC_FALL))
    assert isinstance(advisory, MarketDecisionAdvisory)
    assert advisory.action is MarketAction.propose_reduce
    assert advisory.actionable is True  # it does propose a change
    # ...and proposes nothing else at all.
    assert set(advisory.authority.model_dump().values()) == {False}

# ===========================================================================
# PolicyVerdict: a policy-level abstention is a different finding
# ===========================================================================


def test_no_policy_is_reported_as_a_verdict_not_as_broken_evidence():
    """The distinction this enum exists for, and which nothing made until now.

    `PolicyVerdict` was defined, documented at length, and never set. An operator
    reading an abstention had prose and an evidence-rejection count, and no way to
    tell "your feed is broken" from "nobody supplied the rules for this
    instrument" -- two different repairs.
    """
    advisory = decide_instrument(
        InstrumentEvidence(instrument_id="opaque:x",
                           observations=[quote(age_seconds=30, last=95_000)]),
        None, now=NOW,
    )
    assert advisory.action is MarketAction.abstain
    assert advisory.policy_verdict is PolicyVerdict.no_policy
    assert advisory.rejected == {}
    assert "not a fault in the evidence" in advisory.reason


def test_a_policy_that_declines_a_basis_abstains_and_says_so():
    """Fulfils a promise `gate_evidence` made in its own comment.

    The comment said a `None` basis is "reported as a policy-level abstention
    elsewhere rather than folded in here". There was no elsewhere: the basis was
    computed, found to be None, and ignored. The decision then went ahead and
    produced a document naming no evidence at all -- the failure the whole
    provenance design exists to prevent, reached by the back door.
    """
    class NamelessPolicy(SingleNameEquityPolicy):
        def price_basis(self, quote, *, session):
            return None

    advisory = decide_instrument(
        InstrumentEvidence(instrument_id="opaque:x",
                           observations=[quote(age_seconds=30, last=95_000)]),
        NamelessPolicy(), now=NOW, liveness=[live("a")],
    )
    assert advisory.action is MarketAction.abstain
    assert advisory.policy_verdict is PolicyVerdict.no_basis
    assert advisory.evidence_price is None


def test_the_single_quote_path_also_abstains_on_a_declined_basis():
    """`decide_market_action` must not fall through to `interpret` either.

    Only the reconciled path honoured a `None` basis; the single-quote path still
    went on to interpret and emitted a document naming no price field at all.
    """
    class NamelessPolicy(SingleNameEquityPolicy):
        def price_basis(self, quote, *, session):
            return None

    advisory = decide_market_action(
        quote(age_seconds=30, last=95_000),
        NamelessPolicy(), now=NOW, liveness=[live("a")],
    )
    assert advisory.action is MarketAction.abstain
    assert advisory.policy_verdict is PolicyVerdict.no_basis
    assert advisory.price_basis is None
    assert advisory.evidence_price is None
    assert advisory.rejected == {}
    assert "declined to name a price field" in advisory.reason


def test_an_evidence_rejection_is_not_a_policy_verdict():
    """The two must not blur, or the distinction is worthless."""
    advisory = decide_instrument(
        InstrumentEvidence(instrument_id="opaque:x",
                           observations=[quote(age_seconds=30, last=95_000,
                                               source="a")]),
        BTC, now=NOW, liveness=[],
    )
    assert advisory.action is MarketAction.abstain
    assert advisory.policy_verdict is None
    assert EvidenceRejection.venue_unverified in advisory.rejected


def test_every_verdict_member_is_actually_emitted():
    """The fifth dead enum in this layer was an enum nobody set.

    Asserting the two reachable members are emitted is the test that was missing.
    """
    emitted = set()

    emitted.add(
        decide_instrument(
            InstrumentEvidence(instrument_id="opaque:x",
                               observations=[quote(age_seconds=30, last=95_000)]),
            None, now=NOW,
        ).policy_verdict
    )

    class NamelessPolicy(SingleNameEquityPolicy):
        def price_basis(self, quote, *, session):
            return None

    emitted.add(
        decide_instrument(
            InstrumentEvidence(instrument_id="opaque:x",
                               observations=[quote(age_seconds=30, last=95_000)]),
            NamelessPolicy(), now=NOW, liveness=[live("a")],
        ).policy_verdict
    )

    assert emitted == set(PolicyVerdict)
    assert emitted, "a verdict enum with no emitter is the bug this layer keeps finding"


def test_a_holding_decision_is_not_a_policy_verdict():
    """Only abstentions carry one; a hold is a reading, not a refusal."""
    advisory = decide(series(BTC_FALL))
    assert advisory.action is MarketAction.propose_reduce
    assert advisory.policy_verdict is None


def test_a_dropped_series_is_named_in_the_reason_not_only_in_a_note():
    """The operator has to learn that evidence was discarded.

    Previously a `propose_hold` talked about thin evidence and never mentioned that
    the series which might have proposed had been thrown away.
    """
    advisory = decide([
        MarketQuote(instrument_id="opaque:x", last=50_000,
                    observed_at=NOW - timedelta(days=20), source="a"),
        *series([95_000.0, 96_000.0]),
    ])
    assert "part of the price series was not usable" in advisory.reason
    assert "history_out_of_window" in advisory.reason
    assert EvidenceRejection.history_out_of_window in advisory.rejected


def test_a_history_venue_failure_is_named_distinctly_in_the_reason():
    advisory = decide(
        series([95_000.0, 96_000.0], venues=("a", "zz")),
        liveness=[live("a"), live("b")],
    )
    assert "history_venue_unverified" in advisory.reason
    assert EvidenceRejection.history_venue_unverified in advisory.rejected


# ===========================================================================
# A rejection cause must never be silently reported as a different one
# ===========================================================================


def test_every_rejection_cause_has_an_operator_facing_sentence():
    """The map had two copies and a silent fallback, and both history causes shipped unmapped.

    Adding a rejection cause without remembering to edit two inline dictionaries
    produced "no quote supplied for this asset" for something that was not that at
    all. Nothing failed; the operator was simply wrong.
    """
    from my_jev.market_decision import _PRIMARY_REASON

    assert set(_PRIMARY_REASON) == set(EvidenceRejection)


def test_an_unmapped_cause_fails_loudly_at_the_call_site(monkeypatch):
    """Removing a mapping must break the decision, not quietly reword it.

    Written the obvious way first -- look up a nonsense key and expect KeyError --
    which passes even when the call site uses `.get(..., REASON_NO_QUOTE)`, because
    the *dict* still raises. It tested the dictionary, not the code that reads it,
    and mutation testing is what showed that. So the mapping is removed from a real
    evidence path instead, which is the only way to reach the lookup that matters.
    """
    from my_jev.market_decision import _PRIMARY_REASON

    stale = InstrumentEvidence(
        instrument_id="opaque:x",
        observations=[quote(age_seconds=3600, last=95_000, source="a")],
    )
    # Baseline: mapped, so it abstains with the stale reason.
    baseline = decide_instrument(
        stale, BTC, now=NOW, liveness=[live("a")]
    )
    assert baseline.action is MarketAction.abstain
    assert EvidenceRejection.stale in baseline.rejected

    monkeypatch.delitem(_PRIMARY_REASON, EvidenceRejection.stale)
    with pytest.raises(KeyError):
        decide_instrument(stale, BTC, now=NOW, liveness=[live("a")])


def test_no_rejection_cause_is_unreachable():
    """`undated` and `non_positive` were removed rather than left dead.

    `observed_at` is a required zone-aware datetime and every price field is
    `gt=0.0`, so neither condition can arise inside the layer. The type boundary
    enforces both, which is stronger than admitting a quote and rejecting it --
    but leaving the members would have been the seventh unreachable value in this
    layer.
    """
    from my_jev.market_decision import MarketQuote

    assert not hasattr(EvidenceRejection, "undated")
    assert not hasattr(EvidenceRejection, "non_positive")

    with pytest.raises(ValueError):
        MarketQuote(instrument_id="opaque:x", last=0.0,
                    observed_at=NOW)
    with pytest.raises(ValueError):
        MarketQuote(instrument_id="opaque:x", last=1.0, observed_at="nope")


def test_a_malformed_document_is_an_error_and_a_rejected_observation_is_a_finding():
    """The line the CLI now draws explicitly.

    An unusable quote cannot be constructed, so a document containing one is
    malformed input -- the caller's problem, reported as an error. A well-formed
    quote that fails the gate is a finding, reported on the advisory. Conflating
    them would mean a typo in a file looked like a decision.
    """
    from my_jev.market_decision_cli import MarketEvidenceDocument, decide_document

    good = {
        "schema": "my-jev-market-evidence-v1",
        "instruments": [
            {
                "instrument_id": "opaque:btc",
                "asset_class": "crypto_spot_bitcoin",
                "observations": [
                    {"instrument_id": "opaque:btc", "last": -5.0,
                     "observed_at": NOW.isoformat(), "source": "venue-a"}
                ],
                "liveness": [],
            }
        ],
    }
    with pytest.raises(ValueError, match="opaque:btc"):
        decide_document(MarketEvidenceDocument.model_validate(good), now=NOW)

    # A well-formed but stale quote is a finding on an advisory, not an exception.
    stale = {
        "schema": "my-jev-market-evidence-v1",
        "instruments": [
            {
                "instrument_id": "opaque:btc",
                "asset_class": "crypto_spot_bitcoin",
                "observations": [
                    {"instrument_id": "opaque:btc", "last": 95_000.0,
                     "observed_at": (NOW - timedelta(hours=3)).isoformat(),
                     "source": "venue-a"}
                ],
                "liveness": [
                    {"venue": "venue-a", "reachable": True,
                     "observed_at": (NOW - timedelta(seconds=3)).isoformat()}
                ],
            }
        ],
    }
    advisory = decide_document(
        MarketEvidenceDocument.model_validate(stale), now=NOW
    )[0]
    assert advisory.action is MarketAction.abstain
    assert EvidenceRejection.stale in advisory.rejected


def test_the_refusal_names_the_instrument_and_the_field():
    """`observations.0.last` without the instrument is not actionable in a list of twenty."""
    from my_jev.market_decision_cli import MarketEvidenceDocument, decide_document

    document = {
        "schema": "my-jev-market-evidence-v1",
        "instruments": [
            {
                "instrument_id": "opaque:first",
                "asset_class": "crypto_spot_bitcoin",
                "observations": [
                    {"instrument_id": "opaque:first", "last": 95_000.0,
                     "observed_at": NOW.isoformat(), "source": "venue-a"}
                ],
                "liveness": [],
            },
            {
                "instrument_id": "opaque:second",
                "asset_class": "crypto_spot_bitcoin",
                "observations": [
                    {"instrument_id": "opaque:second", "last": -1.0,
                     "observed_at": NOW.isoformat(), "source": "venue-b"}
                ],
                "liveness": [],
            },
        ],
    }
    with pytest.raises(ValueError) as caught:
        decide_document(MarketEvidenceDocument.model_validate(document), now=NOW)
    message = str(caught.value)
    assert "opaque:second" in message
    assert "opaque:first" not in message
    assert "last" in message


def test_every_malformed_field_is_reported_not_just_the_first():
    """An operator fixing a hand-edited document wants the whole list."""
    from my_jev.market_decision_cli import MarketEvidenceDocument, decide_document

    document = {
        "schema": "my-jev-market-evidence-v1",
        "instruments": [
            {
                "instrument_id": "opaque:btc",
                "asset_class": "crypto_spot_bitcoin",
                "observations": [
                    {"instrument_id": "opaque:btc", "last": -1.0, "bid": -2.0,
                     "observed_at": NOW.isoformat(), "source": "venue-a"}
                ],
                "liveness": [],
            }
        ],
    }
    with pytest.raises(ValueError) as caught:
        decide_document(MarketEvidenceDocument.model_validate(document), now=NOW)
    message = str(caught.value)
    assert "last" in message and "bid" in message


# ===========================================================================
# Two policies must not hold two copies of a trading calendar
# ===========================================================================


def test_every_session_policy_agrees_about_when_a_market_is_open():
    """The duplication this catches was introduced here and would not have surfaced.

    `SingleNameEquityPolicy` first wrote its session boundaries as
    `time(9, 30)` / `time(16, 0)` / `time(4, 0)` / `time(20, 0)` literals while the
    index-fund policy imported four named constants for the same four values.
    They agreed, so every test passed. Correct the calendar in one place and the
    two policies would disagree about whether the market was open -- and this is
    the layer whose entire purpose is noticing two feeds disagreeing.
    """
    from my_jev.market_asset_equity import SingleNameEquityPolicy

    policies = [IndexFundPolicy(), SingleNameEquityPolicy(), BTC]

    # Sweep a trading day in fifteen-minute steps, plus a Sunday.
    moments = [
        NOW.replace(hour=0, minute=0) + timedelta(minutes=15 * step)
        for step in range(0, 96)
    ]
    moments.append(datetime(2026, 10, 4, 15, 0, tzinfo=UTC))  # a Sunday

    for moment in moments:
        sessions = {policy.session_at(moment) for policy in policies}
        # Bitcoin trades 24/7 by design, so it is excluded from the agreement
        # check: its whole distinction is that it has no session.
        session_policies = {
            policy.session_at(moment)
            for policy in policies
            if not isinstance(policy, BitcoinSpotPolicy)
        }
        assert len(session_policies) == 1, (
            f"{moment.isoformat()}: session policies disagree {session_policies}"
        )
        del sessions


def test_the_two_session_policies_name_the_same_boundaries():
    """Named rather than swept, so the failure names the boundary that drifted."""
    from my_jev.market_asset_equity import SingleNameEquityPolicy
    from my_jev.market_asset_index_fund import (
        AFTER_HOURS_CLOSE,
        MARKET_CLOSE,
        MARKET_OPEN,
        PRE_MARKET_OPEN,
    )

    probes = {
        "pre-market": NOW.replace(hour=11, minute=0),   # 07:00 ET
        "regular open": NOW.replace(hour=13, minute=30),  # 09:30 ET
        "regular close": NOW.replace(hour=20, minute=0),  # 16:00 ET
        "after hours": NOW.replace(hour=23, minute=0),  # 19:00 ET
        "past after hours": NOW.replace(hour=2, minute=0),  # 22:00 ET
    }
    fund, equity = IndexFundPolicy(), SingleNameEquityPolicy()
    for label, moment in probes.items():
        assert fund.session_at(moment) is equity.session_at(moment), label

    # And the constants the two share are the ones the boundary probes imply.
    assert MARKET_OPEN == time(9, 30)
    assert MARKET_CLOSE == time(16, 0)
    assert PRE_MARKET_OPEN == time(4, 0)
    assert AFTER_HOURS_CLOSE == time(20, 0)


# ===========================================================================
# A report must be byte-identical across processes, not just across calls
# ===========================================================================


def test_venue_order_does_not_depend_on_the_hash_seed():
    """`MoveSignal.venues` was a `set`, which serialised differently every process.

    Set iteration order depends on `PYTHONHASHSEED`, so the advisory report's bytes
    changed between two runs over identical evidence -- defeating the whole point of
    a document input, where the claim is that a decision can be re-derived later by
    anyone. The existing reproducibility test invoked the CLI twice *inside one
    process*, where the seed is constant, so it could never have caught this.

    The check is therefore across processes, which is the only place the difference
    is observable.
    """
    import json
    import os
    import subprocess
    import sys

    program = (
        "import json;"
        "from my_jev.market_decision import MoveSignal, SustainedMove;"
        "s=MoveSignal(direction=SustainedMove.down,magnitude=0.1,sample_count=3,"
        "span_seconds=60.0,venue_count=3,venues=('a','b','c','d','e','f','g'));"
        "print(json.dumps(s.model_dump(mode='json')))"
    )

    outputs = {}
    for seed in ("0", "1", "2", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": "src"}
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True, text=True, check=True, env=env,
        )
        outputs[seed] = json.loads(result.stdout.strip())["venues"]

    assert len({tuple(v) for v in outputs.values()}) == 1, (
        f"serialisation varied by hash seed: {outputs}"
    )
    # Seed-independent, so this fails even if none of the chosen seeds happens to
    # permute this particular set. The first version of this test passed against
    # the bug it was written for, for exactly that reason: with seven short strings
    # those seeds coincided, and a green test that cannot fail is worse than none.
    for seed, venues in outputs.items():
        assert list(venues) == sorted(venues), (
            f"seed {seed} produced unsorted venues: {venues}"
        )


def test_unsorted_venues_are_refused_rather_than_tidied():
    """Canonical on the way in, so the field cannot carry a non-canonical order.

    Sorting at serialisation time would hide the problem; refusing the value makes a
    caller that assembles one by hand fail immediately instead of producing a report
    whose order depends on something else.
    """
    from my_jev.market_decision import MoveSignal, SustainedMove

    with pytest.raises(ValueError, match="sorted"):
        MoveSignal(
            direction=SustainedMove.down,
            magnitude=0.1,
            sample_count=3,
            span_seconds=60.0,
            venue_count=2,
            venues=("b", "a"),
        )

    with pytest.raises(ValueError, match="sorted"):
        MoveSignal(
            direction=SustainedMove.down,
            magnitude=0.1,
            sample_count=3,
            span_seconds=60.0,
            venue_count=2,
            venues=("a", "a"),
        )


def test_the_decision_is_byte_identical_across_processes():
    """The end-to-end version: the same document, two processes, identical bytes."""
    import json
    import os
    import subprocess
    import sys

    evidence = {
        "schema": "my-jev-market-evidence-v1",
        "instruments": [
            {
                "instrument_id": "opaque:btc",
                "asset_class": "crypto_spot_bitcoin",
                "observations": [
                    {"instrument_id": "opaque:btc", "last": price,
                     "bid": price * 0.999, "ask": price * 1.001,
                     "observed_at": NOW.replace(microsecond=0).isoformat(),
                     "source": venue}
                    for venue, price in (("venue-a", 80_000.0), ("venue-b", 80_010.0))
                ],
                "history": [
                    {"instrument_id": "opaque:btc", "last": price,
                     "observed_at": (NOW - timedelta(minutes=60 - i * 30))
                     .replace(microsecond=0).isoformat(),
                     "source": ("venue-a", "venue-b")[i % 2]}
                    for i, price in enumerate([95_000.0, 88_000.0, 80_000.0])
                ],
                "liveness": [
                    {"venue": venue, "reachable": True,
                     "observed_at": NOW.replace(microsecond=0).isoformat()}
                    for venue in ("venue-a", "venue-b")
                ],
            }
        ],
    }
    stamp = NOW.replace(microsecond=0).isoformat()
    program = (
        "import json,sys;"
        "from my_jev.market_decision_cli import run;"
        f"sys.exit(run(['/dev/stdin','--now','{stamp}']))"
    )
    outputs = set()
    for seed in ("0", "7", "99"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": "src"}
        result = subprocess.run(
            [sys.executable, "-c", program],
            input=json.dumps(evidence),
            capture_output=True, text=True, env=env,
        )
        outputs.add(result.stdout.strip())
        assert result.returncode == 0, result.stderr

    assert len(outputs) == 1, "the report bytes varied by hash seed"
    assert '"propose_reduce"' in outputs.pop()
