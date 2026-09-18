import torch

from my_jev.losses import brier_loss, ordinal_emd_loss


def test_brier_zero_for_exact_target():
    probabilities = torch.tensor([0.0, 1.0])
    target = torch.tensor([0.0, 1.0])
    assert float(brier_loss(probabilities, target)) == 0.0


def test_ordinal_near_miss_costs_less_than_far_miss():
    target = torch.tensor([0.0, 0.0, 1.0, 0.0])
    near = torch.tensor([0.0, 1.0, 0.0, 0.0])
    far = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert ordinal_emd_loss(near, target) < ordinal_emd_loss(
        far,
        target,
    )
