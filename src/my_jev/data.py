from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from torch.utils.data import Dataset

from .schema import DecisionRecord


def load_jsonl(path: str | Path) -> list[DecisionRecord]:
    records: list[DecisionRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(DecisionRecord.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"invalid record at {path}:{line_number}: {exc}") from exc
    return records


def dump_jsonl(records: Iterator[DecisionRecord], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")


class DecisionDataset(Dataset[DecisionRecord]):
    def __init__(self, path: str | Path):
        self.records = load_jsonl(path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> DecisionRecord:
        return self.records[index]


def collate_records(batch: list[DecisionRecord]) -> list[DecisionRecord]:
    return batch
