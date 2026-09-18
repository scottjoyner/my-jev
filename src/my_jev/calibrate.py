from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .calibration import TemperatureScaler
from .checkpoint import load_checkpoint
from .data import DecisionDataset, collate_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit temperature scaling on a held-out calibration split"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, device=device)
    loader = DataLoader(
        DecisionDataset(args.data),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_records,
    )

    logits: list[torch.Tensor] = []
    targets: list[int | torch.Tensor] = []

    model.eval()
    with torch.inference_mode():
        for records in loader:
            for output in model.forward_records(records):
                target = (
                    records[output.record_index].targets or {}
                ).get(output.name)
                if target is None:
                    continue
                logits.append(output.logits.detach())
                if target.distribution is not None:
                    targets.append(
                        torch.tensor(
                            target.distribution,
                            dtype=output.logits.dtype,
                            device=output.logits.device,
                        )
                    )
                else:
                    assert target.index is not None
                    targets.append(target.index)

    if not logits:
        raise SystemExit("calibration split contains no labeled questions")

    scaler = TemperatureScaler().to(device)
    temperature = scaler.fit(logits, targets)

    payload = {
        "temperature": temperature,
        "question_count": len(logits),
        "checkpoint": str(Path(args.checkpoint)),
        "calibration_data": str(Path(args.data)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
