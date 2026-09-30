from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .causal_scalar import (
    CausalScalarSystemOneModel,
)
from .data import (
    DecisionDataset,
    collate_records,
)
from .losses import (
    LossWeights,
    batch_loss,
)
from .train import (
    evaluate,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a LoRA causal scalar "
            "System-One decision model"
        )
    )
    parser.add_argument(
        "--train",
        required=True,
    )
    parser.add_argument(
        "--valid",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--backbone",
        default=(
            "Qwen/Qwen3.5-4B-Base"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--target-modules",
        default="all-linear",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(
        args.seed
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    use_bf16 = bool(
        args.bf16
        and device.type == "cuda"
        and torch.cuda
        .is_bf16_supported()
    )
    if (
        args.bf16
        and not use_bf16
    ):
        print(
            "warning: --bf16 requested "
            "but not reported as supported"
        )

    train_data = DecisionDataset(
        args.train
    )
    valid_data = DecisionDataset(
        args.valid
    )
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

    dtype = (
        torch.bfloat16
        if use_bf16
        else None
    )
    model = (
        CausalScalarSystemOneModel(
            backbone=args.backbone,
            max_length=args.max_length,
            lora_r=args.lora_r,
            lora_alpha=(
                args.lora_alpha
            ),
            lora_dropout=(
                args.lora_dropout
            ),
            target_modules=(
                args.target_modules
            ),
            enable_lora=True,
            device=device,
            dtype=dtype,
        )
    )
    if (
        args.gradient_checkpointing
    ):
        model.gradient_checkpointing_enable()

    parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = AdamW(
        parameters,
        lr=args.lr,
        weight_decay=(
            args.weight_decay
        ),
    )
    weights = LossWeights()

    output = Path(
        args.output
    )
    output.mkdir(
        parents=True,
        exist_ok=True,
    )
    run_config = {
        "model_type": "causal_scalar",
        "backbone": args.backbone,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": (
            args.batch_size
        ),
        "grad_accum": (
            args.grad_accum
        ),
        "lr": args.lr,
        "weight_decay": (
            args.weight_decay
        ),
        "max_length": (
            args.max_length
        ),
        "lora_r": args.lora_r,
        "lora_alpha": (
            args.lora_alpha
        ),
        "lora_dropout": (
            args.lora_dropout
        ),
        "target_modules": (
            args.target_modules
        ),
        "bf16_requested": (
            args.bf16
        ),
        "bf16_enabled": (
            use_bf16
        ),
        "gradient_checkpointing": (
            args.gradient_checkpointing
        ),
        "seed": args.seed,
        "train_records": len(
            train_data
        ),
        "valid_records": len(
            valid_data
        ),
    }
    (
        output
        / "run_config.json"
    ).write_text(
        json.dumps(
            run_config,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_config": (
                    run_config
                )
            },
            indent=2,
        )
    )

    best_nll = float("inf")
    optimizer.zero_grad(
        set_to_none=True
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()
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
                enabled=use_bf16,
            ):
                outputs = (
                    model.forward_records(
                        records
                    )
                )
                loss, metrics = (
                    batch_loss(
                        outputs,
                        records,
                        weights=weights,
                    )
                )
                scaled = (
                    loss
                    / args.grad_accum
                )
            scaled.backward()

            if (
                step
                % args.grad_accum
                == 0
                or step
                == len(
                    train_loader
                )
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
                    "validation": (
                        validation
                    ),
                },
                indent=2,
            )
        )
        model.save_checkpoint(
            output / "last",
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
        if (
            validation["nll"]
            < best_nll
        ):
            best_nll = float(
                validation["nll"]
            )
            model.save_checkpoint(
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
