from __future__ import annotations

import sys

import pytest
import torch

from my_jev.train import (
    NonFiniteTrainingError,
    require_finite_training_step,
)


def test_finite_step_is_allowed():
    require_finite_training_step(
        grad_norm=torch.tensor(0.7),
        loss=torch.tensor(0.21),
        epoch=22,
        step=165,
        lr=2e-5,
    )


def test_non_finite_gradient_stops_the_run():
    with pytest.raises(NonFiniteTrainingError) as excinfo:
        require_finite_training_step(
            grad_norm=torch.tensor(float("nan")),
            loss=torch.tensor(0.21),
            epoch=22,
            step=165,
            lr=2e-5,
        )

    message = str(excinfo.value)
    assert "gradient norm" in message
    assert "epoch 22 step 165" in message
    assert "--lr" in message


def test_non_finite_loss_stops_the_run():
    with pytest.raises(NonFiniteTrainingError) as excinfo:
        require_finite_training_step(
            grad_norm=torch.tensor(0.7),
            loss=torch.tensor(float("inf")),
            epoch=3,
            step=9,
            lr=2e-5,
        )

    assert "loss" in str(excinfo.value)


def test_clipping_alone_would_have_poisoned_the_weights():
    # The reason this guard exists: clip_grad_norm_ scales by the norm, so a
    # NaN gradient survives clipping and NaNs every parameter.
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    parameter.grad = torch.tensor([float("nan"), 1.0])

    torch.nn.utils.clip_grad_norm_([parameter], 1.0)
    assert not torch.isfinite(parameter.grad).all()


def _tiny_split(tmp_path, source, count=6):
    """Write a few real records out of a bundle so main() can run for real."""
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    lines = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            lines.append(line)
            if len(lines) >= count:
                break
    train.write_text("".join(lines), encoding="utf-8")
    valid.write_text("".join(lines[:2]), encoding="utf-8")
    return train, valid


def test_the_training_loop_stops_on_a_non_finite_gradient(
    tmp_path,
    monkeypatch,
):
    """The guard is only useful if the loop actually calls it."""
    from pathlib import Path as _Path

    import my_jev.train as train_module

    bundle = _Path("/tmp/tq-big/my-jev/train.jsonl")
    if not bundle.exists():
        pytest.skip("needs an exported bundle")
    train_file, valid_file = _tiny_split(tmp_path, bundle)

    def _nan_norm(*args, **kwargs):
        return torch.tensor(float("nan"))

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _nan_norm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "my_jev.train",
            "--train",
            str(train_file),
            "--valid",
            str(valid_file),
            "--output",
            str(tmp_path / "out.pt"),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--seed",
            "1",
        ],
    )

    with pytest.raises(NonFiniteTrainingError):
        train_module.main()
