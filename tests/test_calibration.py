import torch

from my_jev.calibration import TemperatureScaler


def test_temperature_scaler_accepts_soft_targets():
    scaler = TemperatureScaler()
    logits = [
        torch.tensor([3.0, 0.0]),
        torch.tensor([0.0, 3.0]),
    ]
    targets = [
        torch.tensor([0.8, 0.2]),
        torch.tensor([0.2, 0.8]),
    ]
    temperature = scaler.fit(logits, targets, max_iter=8)
    assert 0.05 <= temperature <= 20.0


def test_temperature_scaler_accepts_hard_targets():
    scaler = TemperatureScaler()
    temperature = scaler.fit(
        [torch.tensor([0.0, 2.0])],
        [1],
        max_iter=4,
    )
    assert 0.05 <= temperature <= 20.0


def test_temperature_scaler_accepts_inference_mode_tensors():
    with torch.inference_mode():
        logits = [torch.tensor([3.0, 0.0])]
        targets = [torch.tensor([0.7, 0.3])]
    scaler = TemperatureScaler()
    temperature = scaler.fit(logits, targets, max_iter=4)
    assert 0.05 <= temperature <= 20.0
