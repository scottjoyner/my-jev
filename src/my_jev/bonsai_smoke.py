from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .activation_protocol import (
    read_activation_frame,
)
from .external_decision import (
    ExternalActivationDecisionModel,
)


def run_bonsai_smoke(
    frame_path: str | Path,
    *,
    rank: int = 256,
    seed: int = 113,
) -> dict[str, object]:
    source = Path(
        frame_path
    ).expanduser().resolve()

    with source.open(
        "rb"
    ) as handle:
        frame = (
            read_activation_frame(
                handle
            )
        )

    torch.manual_seed(
        seed
    )
    model = (
        ExternalActivationDecisionModel(
            provider=(
                frame.provider
            ),
            hidden_size=(
                frame.hidden_size
            ),
            tap_count=(
                frame.tap_count
            ),
            rank=rank,
            dropout=0.0,
        )
    )
    predictions = (
        model.predict_frame(
            frame
        )
    )

    return {
        "schema_version": 1,
        "kind": (
            "bonsai2-decision-head-smoke"
        ),
        "frame": {
            "path": str(
                source
            ),
            "provider": (
                frame.provider
            ),
            "model_sha256": (
                frame.model_sha256
            ),
            "runtime_revision": (
                frame.runtime_revision
            ),
            "prompt_contract": (
                frame.prompt_contract
            ),
            "hidden_size": (
                frame.hidden_size
            ),
            "tap_count": (
                frame.tap_count
            ),
            "state_tokens": int(
                frame.state_taps.shape[
                    2
                ]
            ),
            "options": int(
                frame.option_taps.shape[
                    1
                ]
            ),
            "questions": len(
                frame.groups
            ),
        },
        "head": {
            "kind": (
                "external_decision_attention"
            ),
            "rank": rank,
            "seed": seed,
            "trained": False,
        },
        "predictions": (
            predictions
        ),
        "quality_claim": {
            "validated": False,
            "reason": (
                "smoke head is randomly "
                "initialized; this proves only "
                "runtime-to-head compatibility"
            ),
        },
        "authority_boundary": {
            "evidence_only": True,
            "dispatch_allowed": False,
            "runtime_authority_changed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run an untrained typed decision "
            "attention head over a captured "
            "Bonsai activation frame"
        )
    )
    parser.add_argument(
        "--frame",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=113,
    )
    args = parser.parse_args()

    result = run_bonsai_smoke(
        args.frame,
        rank=args.rank,
        seed=args.seed,
    )
    output = Path(
        args.output
    )
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    rendered = (
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    output.write_text(
        rendered,
        encoding="utf-8",
    )
    print(
        rendered,
        end="",
    )


if __name__ == "__main__":
    main()
