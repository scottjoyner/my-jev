from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from my_jev.market_decision_cli import (
    MarketEvidenceDocument,
    build_parser,
    decide_document,
    main,
)

NOW = datetime(2026, 10, 6, 15, 0, tzinfo=UTC)


def _document(*blocks):
    return MarketEvidenceDocument(
        instruments=[MarketEvidenceDocument.model_fields["instruments"].annotation.__args__[0](**block) for block in blocks]
    )


def _index_block():
    return {
        "instrument_id": "opaque:index-core",
        "asset_class": "us_equity_index_fund",
        "observations": [
            {
                "instrument_id": "opaque:index-core",
                "last": 500.0,
                "nav": 499.0,
                "bid": 499.9,
                "ask": 500.1,
                "observed_at": (NOW - timedelta(seconds=8)).isoformat(),
                "session": "regular",
                "source": "consolidated-tape",
            }
        ],
        "liveness": [],
    }


def _bitcoin_block(venue_liveness=True, **quote_overrides):
    quote = {
        "instrument_id": "opaque:btc",
        "last": 95_000.0,
        "bid": 94_990.0,
        "ask": 95_010.0,
        "observed_at": (NOW - timedelta(seconds=4)).isoformat(),
        "source": "venue-a",
    }
    quote.update(quote_overrides)
    return {
        "instrument_id": "opaque:btc",
        "asset_class": "crypto_spot_bitcoin",
        "observations": [quote],
        "liveness": [
            {
                "venue": "venue-a",
                "reachable": venue_liveness,
                "observed_at": (NOW - timedelta(seconds=3)).isoformat(),
            }
        ],
    }


def _write(tmp_path, document):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# --- the document ----------------------------------------------------------


def test_the_document_defaults_to_its_schema():
    assert MarketEvidenceDocument().schema == "my-jev-market-evidence-v1"


def test_unknown_fields_are_refused():
    with pytest.raises(ValueError):
        MarketEvidenceDocument.model_validate({"schema": "my-jev-market-evidence-v1", "extra": 1})


def test_a_wrong_schema_is_refused():
    with pytest.raises(ValueError):
        MarketEvidenceDocument.model_validate({"schema": "something-else"})


def test_an_unknown_asset_class_is_refused_rather_than_abstained_on():
    """A typo must not look like a deliberate decision not to reason about it."""
    document = _document({"instrument_id": "opaque:x", "asset_class": "perpetual_swap"})
    with pytest.raises(ValueError, match="unknown asset class"):
        decide_document(document, now=NOW)


# --- deciding --------------------------------------------------------------


def test_every_instrument_is_decided():
    advisories = decide_document(
        _document(_index_block(), _bitcoin_block()), now=NOW
    )
    assert {a.instrument_id for a in advisories} == {"opaque:index-core", "opaque:btc"}


def test_output_order_is_stable_and_sorted():
    """Re-running over the same evidence must reproduce the same document."""
    forward = decide_document(_document(_bitcoin_block(), _index_block()), now=NOW)
    backward = decide_document(_document(_index_block(), _bitcoin_block()), now=NOW)
    assert [a.instrument_id for a in forward] == ["opaque:btc", "opaque:index-core"]
    assert [a.model_dump(mode="json") for a in forward] == [
        a.model_dump(mode="json") for a in backward
    ]


def test_a_dead_venue_is_reported_as_an_abstention():
    advisories = decide_document(_document(_bitcoin_block(venue_liveness=False)), now=NOW)
    assert advisories[0].action.value == "abstain"
    assert "no counterparty" in advisories[0].reason


def test_a_live_venue_decides_normally():
    advisories = decide_document(_document(_bitcoin_block()), now=NOW)
    assert advisories[0].action.value == "propose_hold"


def test_liveness_is_per_instrument_not_global():
    advisories = decide_document(
        _document(_bitcoin_block(venue_liveness=False), _index_block()), now=NOW
    )
    by_id = {a.instrument_id: a for a in advisories}
    assert by_id["opaque:btc"].action.value == "abstain"
    assert by_id["opaque:index-core"].action.value == "propose_hold"


def test_an_empty_document_yields_nothing():
    assert decide_document(MarketEvidenceDocument(), now=NOW) == []


def test_deciding_grants_no_authority():
    for advisory in decide_document(_document(_index_block(), _bitcoin_block()), now=NOW):
        assert set(advisory.authority.model_dump().values()) == {False}
        assert advisory.price_forecast is False


# --- the CLI ---------------------------------------------------------------


def test_cli_reports_every_instrument(tmp_path, capsys):
    path = _write(
        tmp_path,
        {
            "schema": "my-jev-market-evidence-v1",
            "instruments": [_index_block(), _bitcoin_block()],
        },
    )
    assert main([str(path), "--now", NOW.isoformat()]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["instrument_count"] == 2
    assert report["abstained_count"] == 0
    assert report["advisory_only"] is True
    assert {a["instrument_id"] for a in report["advisories"]} == {
        "opaque:index-core",
        "opaque:btc",
    }


def test_cli_exits_two_when_everything_abstained(tmp_path, capsys):
    """A caller can tell "we have a reading" from "we have no reading"."""
    path = _write(
        tmp_path,
        {
            "schema": "my-jev-market-evidence-v1",
            "instruments": [_bitcoin_block(venue_liveness=False)],
        },
    )
    assert main([str(path), "--now", NOW.isoformat()]) == 2
    report = json.loads(capsys.readouterr().out)
    # The abstentions are still the useful output.
    assert report["abstained_count"] == 1
    assert report["advisories"][0]["action"] == "abstain"


def test_cli_writes_to_output(tmp_path, capsys):
    path = _write(
        tmp_path,
        {"schema": "my-jev-market-evidence-v1", "instruments": [_index_block()]},
    )
    out = tmp_path / "report.json"
    assert main([str(path), "--now", NOW.isoformat(), "--output", str(out)]) == 0
    assert capsys.readouterr().out == ""
    assert json.loads(out.read_text())["instrument_count"] == 1


def test_cli_pinning_now_reproduces_a_decision(tmp_path):
    """The point of a document input: the decision can be re-derived."""
    path = _write(
        tmp_path,
        {"schema": "my-jev-market-evidence-v1", "instruments": [_bitcoin_block()]},
    )
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    main([str(path), "--now", NOW.isoformat(), "--output", str(first)])
    main([str(path), "--now", NOW.isoformat(), "--output", str(second)])
    assert first.read_text() == second.read_text()


def test_cli_rejects_a_naive_now(tmp_path):
    path = _write(
        tmp_path,
        {"schema": "my-jev-market-evidence-v1", "instruments": [_index_block()]},
    )
    with pytest.raises(ValueError, match="offset or Z"):
        main([str(path), "--now", "2026-10-06T15:00:00"])


def test_cli_rejects_an_unknown_asset_class(tmp_path):
    path = _write(
        tmp_path,
        {
            "schema": "my-jev-market-evidence-v1",
            "instruments": [{"instrument_id": "opaque:x", "asset_class": "nope"}],
        },
    )
    with pytest.raises(ValueError, match="unknown asset class"):
        main([str(path), "--now", NOW.isoformat()])


def test_cli_carries_the_liveness_window_through(tmp_path, capsys):
    """The window is the CLI's, so the same evidence can be judged more strictly."""
    stale_reading = _bitcoin_block()
    stale_reading["liveness"][0]["observed_at"] = (
        NOW - timedelta(seconds=40)
    ).isoformat()
    path = _write(
        tmp_path,
        {"schema": "my-jev-market-evidence-v1", "instruments": [stale_reading]},
    )
    # 40s passes the 60s default...
    assert main([str(path), "--now", NOW.isoformat()]) == 0
    capsys.readouterr()
    # ...and fails a 30s window, which is the comparison the test is about.
    assert (
        main([str(path), "--now", NOW.isoformat(), "--liveness-max-age-seconds", "30"])
        == 2
    )
    report = json.loads(capsys.readouterr().out)
    assert report["advisories"][0]["rejected"]


def test_the_parser_documents_that_it_places_no_orders():
    help_text = " ".join(build_parser().format_help().split())
    assert "Places no orders" in help_text


# --- proposals through the file path ---------------------------------------
#
# `propose_increase` and `propose_reduce` were unreachable here for two releases
# after they became reachable in the core, because `InstrumentBlock` had no
# `history` field. The fourth dead-value bug in this layer, and the same shape as
# the first: a value nothing can emit. Everything below guards the file path
# specifically, since the core's reachability test says nothing about it.


def _bitcoin_block_with_history(prices=(95_000.0, 88_000.0, 80_000.0), venues=("venue-a", "venue-b")):
    count = len(prices)
    span = (count - 1) * 30
    return {
        "instrument_id": "opaque:btc",
        "asset_class": "crypto_spot_bitcoin",
        "observations": [
            {
                "instrument_id": "opaque:btc",
                "last": 80_000.0,
                "bid": 79_920.0,
                "ask": 80_080.0,
                "observed_at": (NOW - timedelta(seconds=4)).isoformat(),
                "source": "venue-a",
            },
            {
                "instrument_id": "opaque:btc",
                "last": 80_010.0,
                "bid": 79_930.0,
                "ask": 80_090.0,
                "observed_at": (NOW - timedelta(seconds=4)).isoformat(),
                "source": "venue-b",
            },
        ],
        "history": [
            {
                "instrument_id": "opaque:btc",
                "last": price,
                "observed_at": (
                    NOW - timedelta(minutes=span - index * 30)
                ).isoformat(),
                "source": venues[index % len(venues)],
            }
            for index, price in enumerate(prices)
        ],
        "liveness": [
            {
                "venue": venue,
                "reachable": True,
                "observed_at": (NOW - timedelta(seconds=3)).isoformat(),
            }
            for venue in venues
        ],
    }


def test_a_sustained_move_in_a_document_proposes():
    """The point of adding `history`: this used to be unreachable from a file."""
    advisory = decide_document(
        MarketEvidenceDocument.model_validate(
            {"schema": "my-jev-market-evidence-v1",
             "instruments": [_bitcoin_block_with_history()]}
        ),
        now=NOW,
    )[0]
    assert advisory.action.value == "propose_reduce"
    assert advisory.move.direction.value == "down"


def test_a_document_without_history_still_holds():
    """The conservative default must survive the new field's arrival."""
    advisory = decide_document(
        MarketEvidenceDocument.model_validate(
            {"schema": "my-jev-market-evidence-v1", "instruments": [_bitcoin_block()]}
        ),
        now=NOW,
    )[0]
    assert advisory.action.value == "propose_hold"
    assert advisory.move is None


def test_history_is_a_separate_field_not_a_smuggled_observation():
    """Two venues quoting at one moment is reconciliation; a series is state change.

    Merged into one list a caller could not say which was which, and a two-hour-old
    print would be reconciled against a fresh one as though both were current.
    """
    block = _bitcoin_block_with_history()
    assert "history" in block
    assert all(entry.get("observed_at") for entry in block["history"])
    # The core refuses a history entry that names another instrument, so a caller
    # cannot smuggle a different instrument's prints into this one's series.
    from my_jev.market_decision import InstrumentEvidence

    with pytest.raises(ValueError):
        InstrumentEvidence.model_validate(
            {
                "instrument_id": "opaque:btc",
                "history": [
                    {"instrument_id": "opaque:other", "last": 1.0,
                     "observed_at": NOW.isoformat()}
                ],
            }
        )


def test_the_thin_equity_policy_is_reachable_from_a_document():
    """A policy that exists but is not registered is a policy nobody can use."""
    advisory = decide_document(
        MarketEvidenceDocument.model_validate(
            {
                "schema": "my-jev-market-evidence-v1",
                "instruments": [
                    {
                        "instrument_id": "opaque:thin",
                        "asset_class": "single_name_equity",
                        "observations": [
                            {"instrument_id": "opaque:thin", "last": 130.0,
                             "bid": 129.87, "ask": 130.13,
                             "observed_at": (NOW - timedelta(seconds=4)).isoformat(),
                             "source": "venue-a"},
                            {"instrument_id": "opaque:thin", "last": 130.02,
                             "bid": 129.89, "ask": 130.15,
                             "observed_at": (NOW - timedelta(seconds=4)).isoformat(),
                             "source": "venue-b"},
                        ],
                        "history": [
                            {"instrument_id": "opaque:thin", "last": price,
                             "observed_at": (
                                 NOW - timedelta(minutes=60 - index * 30)
                             ).isoformat(),
                             "source": ("venue-a", "venue-b")[index % 2]}
                            for index, price in enumerate([100.0, 112.0, 130.0])
                        ],
                        "liveness": [
                            {"venue": venue, "reachable": True,
                             "observed_at": (NOW - timedelta(seconds=3)).isoformat()}
                            for venue in ("venue-a", "venue-b")
                        ],
                    }
                ],
            }
        ),
        now=NOW,
    )[0]
    assert advisory.action.value == "propose_increase"


def test_every_registered_policy_produces_a_decision():
    """A registered policy that always abstains is a registration, not a policy."""
    from my_jev.market_decision_cli import POLICIES

    assert len(POLICIES) == 3
    for asset_class in POLICIES:
        block = _bitcoin_block()
        block["asset_class"] = asset_class
        advisory = decide_document(
            MarketEvidenceDocument.model_validate(
                {"schema": "my-jev-market-evidence-v1", "instruments": [block]}
            ),
            now=NOW,
        )[0]
        assert advisory.action.value != "abstain", asset_class


def test_the_report_names_how_many_instruments_were_proposed(tmp_path):
    """So a reader need not count actions to tell a proposal from a reading.

    Also asserts the count is *zero* on evidence that only holds, which is the half
    that matters: a report whose proposed_count is quietly wrong in the flattering
    direction would be worse than one without the field.
    """
    source = tmp_path / "evidence.json"
    source.write_text(
        json.dumps({
            "schema": "my-jev-market-evidence-v1",
            "instruments": [_bitcoin_block(), _index_block()],
        }),
        encoding="utf-8",
    )
    destination = tmp_path / "report.json"
    exit_code = main([
        str(source),
        "--now", NOW.isoformat(),
        "--output", str(destination),
    ])

    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["instrument_count"] == 2
    assert report["proposed_count"] == 0
    assert report["advisory_only"] is True
    assert exit_code == 0


def test_the_report_counts_a_real_proposal(tmp_path):
    """The other half of the same field: it must count up as well as down."""
    source = tmp_path / "evidence.json"
    source.write_text(
        json.dumps({
            "schema": "my-jev-market-evidence-v1",
            "instruments": [
                _bitcoin_block_with_history(),
                _index_block(),
            ],
        }),
        encoding="utf-8",
    )
    destination = tmp_path / "report.json"
    main([
        str(source),
        "--now", NOW.isoformat(),
        "--output", str(destination),
    ])

    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["proposed_count"] == 1
    assert report["instrument_count"] == 2
    # Provenance has to survive serialisation, or the file a person reviews cannot
    # be checked against the move that justified it.
    proposed = [
        a for a in report["advisories"] if a["action"] == "propose_reduce"
    ]
    assert len(proposed) == 1
    assert proposed[0]["move"]["direction"] == "down"
    assert proposed[0]["move"]["sample_count"] == 3


def test_a_replayed_document_reproduces_the_proposal(tmp_path, capsys):
    """The property this entry point exists for: re-derivable from the same bytes.

    If pinning ``--now`` reproduced a hold but not a proposal, then the proposal
    was not a function of the stored evidence and the audit trail was decorative.
    """
    source = tmp_path / "evidence.json"
    source.write_text(
        json.dumps({
            "schema": "my-jev-market-evidence-v1",
            "instruments": [_bitcoin_block_with_history()],
        }),
        encoding="utf-8",
    )
    main([str(source), "--now", NOW.isoformat()])
    first = capsys.readouterr().out
    main([str(source), "--now", NOW.isoformat()])
    second = capsys.readouterr().out
    assert first == second
    assert json.loads(first)["proposed_count"] == 1


# --- the executable reports a refusal instead of raising a traceback -------


def test_the_executable_reports_a_malformed_document_cleanly(tmp_path, capsys):
    """As a shell command, a bad field used to be a pydantic traceback.

    Which told the operator nothing about whether their evidence was wrong or the
    tool was -- the ambiguity this module's whole argument is against.
    """
    from my_jev.market_decision_cli import run

    path = tmp_path / "evidence.json"
    path.write_text(
        json.dumps({
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
        }),
        encoding="utf-8",
    )
    exit_code = run([str(path), "--now", NOW.isoformat()])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "error:" in captured.err
    assert "opaque:btc" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_the_executable_reports_an_unknown_asset_class_cleanly(tmp_path, capsys):
    """A refusal the CLI already made on purpose, which also dumped a traceback."""
    from my_jev.market_decision_cli import run

    path = tmp_path / "evidence.json"
    path.write_text(
        json.dumps({
            "schema": "my-jev-market-evidence-v1",
            "instruments": [{"instrument_id": "opaque:x", "asset_class": "nope"}],
        }),
        encoding="utf-8",
    )
    exit_code = run([str(path), "--now", NOW.isoformat()])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "unknown asset class" in captured.err
    assert "Traceback" not in captured.err


def test_the_library_still_raises_so_an_embedding_caller_sees_the_error(tmp_path):
    """`main` raising and `run` reporting is a deliberate split, not an oversight."""
    from my_jev.market_decision_cli import main as library_main

    path = tmp_path / "evidence.json"
    path.write_text(
        json.dumps({
            "schema": "my-jev-market-evidence-v1",
            "instruments": [{"instrument_id": "opaque:x", "asset_class": "nope"}],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown asset class"):
        library_main([str(path), "--now", NOW.isoformat()])


def test_the_document_no_longer_warns_on_every_invocation(tmp_path, capsys):
    """A field named `schema` shadowed a `BaseModel` attribute, warning every run.

    A tool that warns on every invocation teaches its operator to ignore warnings,
    and this project's whole position is that the warnings are the output.
    """
    import warnings

    from my_jev.market_decision_cli import MarketEvidenceDocument

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        document = MarketEvidenceDocument.model_validate(
            {"schema": "my-jev-market-evidence-v1", "instruments": []}
        )
    assert document.schema == "my-jev-market-evidence-v1"


def test_the_wire_key_is_still_schema():
    """Renamed internally so the document contract did not change."""
    from my_jev.market_decision_cli import MarketEvidenceDocument

    with pytest.raises(ValueError):
        MarketEvidenceDocument.model_validate(
            {"wrong_key": "my-jev-market-evidence-v1", "instruments": []}
        )
    assert MarketEvidenceDocument.model_fields["document_schema"].validation_alias == "schema"
