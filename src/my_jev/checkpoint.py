from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .model import SystemOneModel


def save_checkpoint(
    model: SystemOneModel,
    output_dir: str | Path,
    *,
    extra: dict | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )
    config = {
        "model_type": (
            "encoder_option_query"
        ),
        "backbone": model.backbone_name,
        "max_state_length": (
            model.max_state_length
        ),
        "max_candidate_length": (
            model.max_candidate_length
        ),
        "head_kind": model.head_kind,
        "head_rank": model.head_rank,
        "extra": extra or {},
    }
    (
        output
        / "my_jev_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    torch.save(
        model.state_dict(),
        output / "model.pt",
    )
    model.tokenizer.save_pretrained(
        output / "tokenizer"
    )


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> Any:
    root = Path(path)
    config = json.loads(
        (
            root
            / "my_jev_config.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    model_type = config.get(
        "model_type",
        "encoder_option_query",
    )
    if (
        model_type
        == "causal_scalar"
    ):
        from .causal_scalar import (
            load_causal_checkpoint,
        )

        return load_causal_checkpoint(
            root,
            device=device,
        )
    if (
        model_type
        != "encoder_option_query"
    ):
        raise ValueError(
            "unsupported checkpoint "
            f"model_type: {model_type}"
        )

    model = SystemOneModel(
        backbone=config[
            "backbone"
        ],
        max_state_length=config[
            "max_state_length"
        ],
        max_candidate_length=config[
            "max_candidate_length"
        ],
        head_kind=config.get(
            "head_kind",
            "legacy",
        ),
        head_rank=config.get(
            "head_rank"
        ),
    )
    state = torch.load(
        root / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(
        state
    )
    model.to(device)
    return model
