from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import save_checkpoint
from .data import DecisionDataset, collate_records
from .losses import LossWeights, batch_loss
from .metrics import multiclass_metrics
from .model import SystemOneModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a typed System-One decision model"
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--backbone",
        default="answerdotai/ModernBERT-base",
    )
    parser.add_argument(
        "--head-kind",
        choices=("option_query", "legacy"),
        default="option_query",
    )
    parser.add_argument(
        "--head-rank",
        type=int,
        default=None,
        help="Projection rank for the option-query head",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--max-state-length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--max-candidate-length",
        type=int,
        default=192,
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help=(
            "Trade compute for lower activation memory "
            "in the backbone"
        ),
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: SystemOneModel,
    loader: DataLoader,
) -> dict[str, float]:
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[int | np.ndarray] = []
    losses = []

    with torch.inference_mode():
        for records in loader:
            outputs = model.forward_records(records)
            loss, _ = batch_loss(
                outputs,
                records,
            )
            losses.append(
                float(loss.item())
            )
            for output in outputs:
                target_map = (
                    records[
                        output.record_index
                    ].targets
                    or {}
                )
                target = target_map.get(
                    output.name
                )
                if target is None:
                    continue

                probabilities.append(
                    output.probabilities
                    .detach()
                    .cpu()
                    .numpy()
                )
                if target.distribution is not None:
                    targets.append(
                        np.asarray(
                            target.distribution,
                            dtype=np.float64,
                        )
                    )
                else:
                    assert target.index is not None
                    targets.append(
                        target.index
                    )

    metrics = multiclass_metrics(
        probabilities,
        targets,
    )
    return {
        "loss": (
            float(np.mean(losses))
            if losses
            else 0.0
        ),
        "accuracy": metrics.accuracy,
        "nll": metrics.nll,
        "brier": metrics.brier,
        "ece": metrics.ece,
        "count": float(metrics.count),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_data = DecisionDataset(args.train)
    valid_data = DecisionDataset(args.valid)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_records,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_records,
    )

    model = SystemOneModel(
        backbone=args.backbone,
        max_state_length=args.max_state_length,
        max_candidate_length=args.max_candidate_length,
        head_kind=args.head_kind,
        head_rank=args.head_rank,
    ).to(device)

    if args.gradient_checkpointing:
        enable = getattr(
            model.encoder,
            "gradient_checkpointing_enable",
            None,
        )
        if enable is None:
            raise SystemExit(
                "selected backbone does not expose "
                "gradient checkpointing"
            )
        enable()

    if args.freeze_backbone:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = AdamW(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    weights = LossWeights()
    use_amp = bool(
        args.bf16
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    if args.bf16 and not use_amp:
        print(
            "warning: --bf16 requested but not "
            "reported as supported; continuing "
            "without autocast"
        )

    best_nll = float("inf")
    output = Path(args.output)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_config = {
        "backbone": args.backbone,
        "head_kind": args.head_kind,
        "head_rank": model.head_rank,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_state_length": args.max_state_length,
        "max_candidate_length": (
            args.max_candidate_length
        ),
        "bf16_requested": args.bf16,
        "bf16_enabled": use_amp,
        "freeze_backbone": (
            args.freeze_backbone
        ),
        "gradient_checkpointing": (
            args.gradient_checkpointing
        ),
        "seed": args.seed,
        "train_records": len(train_data),
        "valid_records": len(valid_data),
    }
    (output / "run_config.json").write_text(
        json.dumps(
            run_config,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"run_config": run_config},
            indent=2,
        )
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()
        optimizer.zero_grad(
            set_to_none=True
        )
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch}",
        )
        for step, records in enumerate(
            progress,
            start=1,
        ):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                outputs = (
                    model.forward_records(
                        records
                    )
                )
                loss, metrics = batch_loss(
                    outputs,
                    records,
                    weights=weights,
                )
                scaled_loss = (
                    loss
                    / args.grad_accum
                )
            scaled_loss.backward()

            if (
                step % args.grad_accum == 0
                or step == len(train_loader)
            ):
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    1.0,
                )
                optimizer.step()
                optimizer.zero_grad(
                    set_to_none=True
                )
            progress.set_postfix(
                loss=(
                    f"{metrics['loss']:.4f}"
                )
            )

        validation = evaluate(
            model,
            valid_loader,
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "validation": validation,
                },
                indent=2,
            )
        )
        save_checkpoint(
            model,
            output / "last",
            extra={
                "epoch": epoch,
                "validation": validation,
                "run_config": run_config,
            },
        )
        if validation["nll"] < best_nll:
            best_nll = validation["nll"]
            save_checkpoint(
                model,
                output / "best",
                extra={
                    "epoch": epoch,
                    "validation": (
                        validation
                    ),
                    "run_config": (
                        run_config
                    ),
                },
            )


if __name__ == "__main__":
    main()
