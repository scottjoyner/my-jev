from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .schema import DecisionRecord


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run typed decisions with a my-jev checkpoint"
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    decide = subparsers.add_parser("decide")
    decide.add_argument("--checkpoint", required=True)
    decide.add_argument("--input", required=True, help="JSON request file")
    args = parser.parse_args()

    if args.command == "decide":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = load_checkpoint(args.checkpoint, device=device)
        payload = json.loads(
            Path(args.input).read_text(encoding="utf-8")
        )
        record = DecisionRecord.model_validate(payload)
        decisions = model.predict([record])
        print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
