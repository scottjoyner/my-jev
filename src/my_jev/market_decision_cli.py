"""Read market evidence from a document and report what it can support.

Advisory only, like every other entry point here: it reads files the caller names,
places no orders, and grants no authority.

The input is a document rather than a live feed, deliberately. This layer's value
is that a decision can be re-derived from the exact evidence that produced it,
months later, by anyone. A tool that quietly polls an exchange cannot do that, and
a decision whose input has to be taken on trust is not one anybody can review.

Venue liveness is part of the input for the same reason it is part of the gate:
"the quote looked fine" and "there was somebody there to trade it against" are
different facts, and only the first is visible in a price.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .market_asset_crypto import BitcoinSpotPolicy
from .market_asset_equity import SingleNameEquityPolicy
from .market_asset_index_fund import IndexFundPolicy
from .market_decision import (
    DEFAULT_LIVENESS_MAX_AGE_SECONDS,
    AssetPolicy,
    InstrumentEvidence,
    MarketDecisionAdvisory,
    VenueLiveness,
    decide_instrument,
)

SCHEMA = "my-jev-market-evidence-v1"

#: Asset classes this build knows how to reason about. An unrecognised one is an
#: error rather than a silent abstention: quietly abstaining on a typo would look
#: identical to a deliberate decision not to, and only one of those is intended.
POLICIES: dict[str, type[AssetPolicy]] = {
    BitcoinSpotPolicy().asset_class: BitcoinSpotPolicy,
    IndexFundPolicy().asset_class: IndexFundPolicy,
    SingleNameEquityPolicy().asset_class: SingleNameEquityPolicy,
}


class InstrumentBlock(BaseModel):
    """Every observation of one instrument, plus the liveness of its venues.

    ``history`` is a separate field from ``observations`` for the same reason it is
    on :class:`InstrumentEvidence`: two venues quoting at one moment is a
    reconciliation problem, and a series of prints over time is a state-change
    problem. Merged into one list, a caller could not express which was which.

    It was missing entirely until now, which made every proposal unreachable
    through this entry point -- the same dead-value bug as the one that left
    ``propose_increase`` unemittable in the core, in a different place. A file that
    a person is asked to review has to be able to say what it is.
    """

    model_config = ConfigDict(extra="forbid")

    instrument_id: str = Field(min_length=1, max_length=64)
    asset_class: str = Field(min_length=1, max_length=64)
    observations: list[dict] = Field(default_factory=list)
    history: list[dict] = Field(default_factory=list)
    liveness: list[dict] = Field(default_factory=list)


class MarketEvidenceDocument(BaseModel):
    """The input a decision is derived from, in full."""

    model_config = ConfigDict(extra="forbid")

    schema: str = SCHEMA
    instruments: list[InstrumentBlock] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_schema(self) -> MarketEvidenceDocument:
        if self.schema != SCHEMA:
            raise ValueError(
                f"unsupported evidence schema {self.schema!r}; this build reads "
                f"{SCHEMA}. Refusing rather than guessing at the shape."
            )
        return self


def _policy_for(asset_class: str) -> AssetPolicy:
    factory = POLICIES.get(asset_class)
    if factory is None:
        raise ValueError(
            f"unknown asset class {asset_class!r}; known: {sorted(POLICIES)}. "
            "Refusing rather than abstaining, so a typo cannot look like a "
            "deliberate decision not to reason about it."
        )
    return factory()


def decide_document(
    document: MarketEvidenceDocument,
    *,
    now=None,
    liveness_max_age_seconds: int = DEFAULT_LIVENESS_MAX_AGE_SECONDS,
) -> list[MarketDecisionAdvisory]:
    """Decide every instrument in the document, in a stable order.

    Sorted by instrument id so re-running over the same evidence produces the same
    document, which is the whole point of keeping the input on disk.
    """
    from datetime import UTC, datetime

    moment = now or datetime.now(UTC)
    out: list[MarketDecisionAdvisory] = []
    for block in sorted(document.instruments, key=lambda item: item.instrument_id):
        policy = _policy_for(block.asset_class)
        evidence = InstrumentEvidence.model_validate(
            {
                "instrument_id": block.instrument_id,
                "observations": block.observations,
                "history": block.history,
            }
        )
        liveness = [VenueLiveness.model_validate(item) for item in block.liveness]
        out.append(
            decide_instrument(
                evidence,
                policy,
                now=moment,
                liveness=liveness,
                liveness_max_age_seconds=liveness_max_age_seconds,
            )
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report what a set of market evidence can support, per instrument. "
            "Places no orders and grants no authority."
        )
    )
    parser.add_argument("evidence", type=Path, help=f"{SCHEMA} JSON document.")
    parser.add_argument(
        "--now",
        help=(
            "Evaluation moment, ISO-8601. Pin it to re-derive a decision from the "
            "evidence that produced it."
        ),
    )
    parser.add_argument(
        "--liveness-max-age-seconds",
        type=int,
        default=DEFAULT_LIVENESS_MAX_AGE_SECONDS,
        help="How recently a venue must have been seen reachable.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def _time(value: str | None):
    from datetime import datetime

    if value is None:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("--now must include an offset or Z")
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document = MarketEvidenceDocument.model_validate_json(
        args.evidence.read_text(encoding="utf-8")
    )
    advisories = decide_document(
        document,
        now=_time(args.now),
        liveness_max_age_seconds=args.liveness_max_age_seconds,
    )

    abstained = [a for a in advisories if a.action.value == "abstain"]
    report = {
        "schema_version": "my-jev-market-decision-report-v1",
        "advisory_only": True,
        "evaluated_at": advisories[0].evaluated_at if advisories else None,
        "instrument_count": len(advisories),
        "abstained_count": len(abstained),
        # Named so a reader does not have to count actions to tell "we proposed
        # something" from "we had a reading". Still only a count: this report is
        # advisory output, and nothing in it places an order.
        "proposed_count": len(advisories) - len(abstained)
        - len([a for a in advisories if a.action.value == "propose_hold"]),
        "advisories": [advisory.model_dump(mode="json") for advisory in advisories],
    }

    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)

    # Non-zero when nothing was decidable, so a caller can tell "we have a reading"
    # from "we have no reading" without parsing the document. Every advisory is
    # emitted either way; the abstentions are the useful output when nothing was
    # decidable.
    return 0 if advisories and len(abstained) < len(advisories) else 2


if __name__ == "__main__":
    sys.exit(main())