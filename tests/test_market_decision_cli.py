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
