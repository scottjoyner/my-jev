from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .locking import atomic_write_json


@dataclass
class ExperimentEntry:
    run_id: str
    experiment: str
    created_at: float
    updated_at: float
    status: str
    spec_path: str
    spec_sha256: str
    model_backend: str
    backbone: str
    train_sha256: str
    validation_sha256: str
    calibration_sha256: str
    test_sha256: str
    run_dir: str
    checkpoint_path: str | None = None
    calibration_path: str | None = None
    benchmark_path: str | None = None
    promotion_path: str | None = None
    promoted: bool = False
    parent_run_id: str | None = None
    notes: list[str] = field(
        default_factory=list
    )
    metadata: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )


class ExperimentRegistry:
    def __init__(
        self,
        path: str | Path,
    ) -> None:
        self.path = Path(path)
        self.entries: dict[
            str,
            ExperimentEntry,
        ] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        import json

        payload = json.loads(
            self.path.read_text(
                encoding="utf-8"
            )
        )
        for run_id, raw in payload.items():
            self.entries[
                run_id
            ] = ExperimentEntry(
                **raw
            )

    def _save(self) -> None:
        atomic_write_json(
            self.path,
            {
                run_id: asdict(
                    entry
                )
                for run_id, entry
                in sorted(
                    self.entries.items()
                )
            },
        )

    def register(
        self,
        entry: ExperimentEntry,
    ) -> ExperimentEntry:
        if (
            entry.run_id
            in self.entries
        ):
            raise ValueError(
                "run already registered: "
                f"{entry.run_id}"
            )
        self.entries[
            entry.run_id
        ] = entry
        self._save()
        return entry

    def update(
        self,
        run_id: str,
        **changes: Any,
    ) -> ExperimentEntry:
        if run_id not in self.entries:
            raise KeyError(
                f"unknown run: {run_id}"
            )
        entry = self.entries[
            run_id
        ]
        for name, value in (
            changes.items()
        ):
            if not hasattr(
                entry,
                name,
            ):
                raise AttributeError(
                    name
                )
            setattr(
                entry,
                name,
                value,
            )
        entry.updated_at = time.time()
        self._save()
        return entry

    def get(
        self,
        run_id: str,
    ) -> ExperimentEntry | None:
        return self.entries.get(
            run_id
        )

    def latest(
        self,
        experiment: str,
        *,
        promoted_only: bool = False,
    ) -> ExperimentEntry | None:
        matches = [
            entry
            for entry in (
                self.entries.values()
            )
            if (
                entry.experiment
                == experiment
                and (
                    not promoted_only
                    or entry.promoted
                )
            )
        ]
        if not matches:
            return None
        return max(
            matches,
            key=lambda entry: (
                entry.created_at
            ),
        )

    def lineage(
        self,
        run_id: str,
    ) -> list[ExperimentEntry]:
        chain: list[
            ExperimentEntry
        ] = []
        seen: set[str] = set()
        current = self.get(
            run_id
        )
        while current is not None:
            if current.run_id in seen:
                raise ValueError(
                    "registry lineage cycle "
                    f"at {current.run_id}"
                )
            seen.add(
                current.run_id
            )
            chain.append(
                current
            )
            current = (
                self.get(
                    current.parent_run_id
                )
                if current.parent_run_id
                else None
            )
        return chain
