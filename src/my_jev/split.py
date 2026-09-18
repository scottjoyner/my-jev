from __future__ import annotations

import argparse
import random
from pathlib import Path

from .data import dump_jsonl, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create train/calibration/test splits "
            "at the record level"
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.train_fraction <= 0 or args.calibration_fraction < 0:
        raise SystemExit("invalid split fractions")
    if args.train_fraction + args.calibration_fraction >= 1:
        raise SystemExit(
            "train + calibration fractions must be < 1"
        )

    records = load_jsonl(args.input)
    random.Random(args.seed).shuffle(records)

    train_end = int(
        len(records) * args.train_fraction
    )
    calibration_end = train_end + int(
        len(records) * args.calibration_fraction
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dump_jsonl(
        iter(records[:train_end]),
        output / "train.jsonl",
    )
    dump_jsonl(
        iter(records[train_end:calibration_end]),
        output / "calibration.jsonl",
    )
    dump_jsonl(
        iter(records[calibration_end:]),
        output / "test.jsonl",
    )


if __name__ == "__main__":
    main()
