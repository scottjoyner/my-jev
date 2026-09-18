from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from .calibration import load_temperature
from .checkpoint import load_checkpoint
from .data import DecisionDataset, collate_records
from .metrics import multiclass_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a my-jev checkpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--calibration",
        help="JSON artifact created by my-jev-calibrate",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = load_checkpoint(
        args.checkpoint,
        device=device,
    )
    temperature = load_temperature(args.calibration)
    loader = DataLoader(
        DecisionDataset(args.data),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_records,
    )

    probabilities: list[np.ndarray] = []
    targets: list[int] = []
    model.eval()
    with torch.inference_mode():
        for records in loader:
            outputs = model.forward_records(records)
            for output in outputs:
                target = (
                    records[output.record_index].targets or {}
                ).get(output.name)
                if target is None:
                    continue
                index = target.index
                if index is None:
                    index = int(
                        np.argmax(target.distribution)
                    )
                probabilities.append(
                    output.probabilities_at_temperature(
                        temperature
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                targets.append(index)

    metrics = multiclass_metrics(
        probabilities,
        targets,
    )
    payload = {
        **metrics.__dict__,
        "temperature": temperature,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
