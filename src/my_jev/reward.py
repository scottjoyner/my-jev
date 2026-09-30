from __future__ import annotations

import torch
from torch import Tensor


def expected_reward_loss(
    logits: Tensor,
    rewards: Tensor,
    *,
    entropy_bonus: float = 0.0,
) -> Tensor:
    """Low-variance objective when a verifier can score every legal option."""
    if logits.shape != rewards.shape:
        raise ValueError("logits and rewards must have identical shape")
    probabilities = torch.softmax(logits, dim=-1)
    expected_reward = torch.sum(probabilities * rewards)
    entropy = -torch.sum(
        probabilities * torch.log(probabilities.clamp_min(1e-9))
    )
    return -expected_reward - entropy_bonus * entropy


def sampled_policy_gradient_loss(
    logits: Tensor,
    sampled_index: int,
    reward: float,
    *,
    baseline: float = 0.0,
) -> Tensor:
    """REINFORCE-style fallback when only the sampled action can be verified."""
    log_probabilities = torch.log_softmax(logits, dim=-1)
    advantage = float(reward - baseline)
    return -log_probabilities[sampled_index] * advantage
