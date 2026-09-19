from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn


class TemperatureScaler(nn.Module):
    """Single positive temperature fit on held-out logits."""

    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.tensor(initial_temperature).log()
        )

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp().clamp(0.05, 20.0)

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature

    def fit(
        self,
        logits: list[Tensor],
        targets: list[int | Tensor],
        max_iter: int = 100,
    ) -> float:
        if not logits:
            raise ValueError("no logits supplied")
        if len(logits) != len(targets):
            raise ValueError("logits and targets must have the same length")

        optimizer = torch.optim.LBFGS(
            [self.log_temperature],
            lr=0.05,
            max_iter=max_iter,
        )
        detached = [item.detach() for item in logits]

        def closure() -> Tensor:
            optimizer.zero_grad()
            losses = []
            for item, target in zip(detached, targets, strict=True):
                scaled = self(item)
                if isinstance(target, int):
                    target_tensor = torch.tensor(
                        [target],
                        device=item.device,
                    )
                    loss = torch.nn.functional.cross_entropy(
                        scaled.unsqueeze(0),
                        target_tensor,
                    )
                else:
                    target_tensor = target.detach().to(
                        device=item.device,
                        dtype=scaled.dtype,
                    )
                    target_tensor = (
                        target_tensor
                        / target_tensor.sum().clamp_min(1e-12)
                    )
                    loss = -(
                        target_tensor
                        * torch.log_softmax(scaled, dim=-1)
                    ).sum()
                losses.append(loss)
            loss = torch.stack(losses).mean()
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.detach().item())


def load_temperature(path: str | Path | None) -> float:
    if path is None:
        return 1.0
    payload = json.loads(
        Path(path).read_text(encoding="utf-8")
    )
    temperature = float(payload["temperature"])
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return temperature
