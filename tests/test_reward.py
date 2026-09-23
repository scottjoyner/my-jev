import torch

from my_jev.reward import expected_reward_loss


def test_expected_reward_prefers_rewarded_option():
    bad = torch.tensor([4.0, 0.0], requires_grad=True)
    good = torch.tensor([0.0, 4.0], requires_grad=True)
    rewards = torch.tensor([0.0, 1.0])
    assert expected_reward_loss(good, rewards) < expected_reward_loss(
        bad,
        rewards,
    )
