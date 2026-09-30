from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from .data import load_jsonl
from .schema import DecisionRecord
from .teacher import (
    OpenAICompatibleTeacher,
    TeacherConfig,
    aggregate_teacher_targets,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Label unlabeled decision records "
            "with an OpenAI-compatible teacher"
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--api-key-env")
    args = parser.parse_args()

    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")

    teacher = OpenAICompatibleTeacher(
        TeacherConfig(
            endpoint=args.endpoint,
            model=args.model,
            temperature=args.temperature,
            api_key_env=args.api_key_env,
        )
    )
    records = load_jsonl(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in tqdm(records, desc="teacher labeling"):
            unlabeled = DecisionRecord(
                state=record.state,
                questions=record.questions,
                metadata=record.metadata,
            )
            samples = [
                teacher.label(unlabeled)
                for _ in range(args.samples)
            ]
            targets = aggregate_teacher_targets(
                unlabeled,
                samples,
            )
            labeled = DecisionRecord(
                state=record.state,
                questions=record.questions,
                targets=targets,
                metadata={
                    **record.metadata,
                    "teacher_model": args.model,
                    "teacher_samples": args.samples,
                },
            )
            handle.write(
                json.dumps(
                    labeled.model_dump(mode="json"),
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
