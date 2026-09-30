import pytest
import torch

from my_jev.determinism import (
    CUBLAS_WORKSPACE_CONFIG,
    CUBLAS_WORKSPACE_CONFIG_ENV,
    configure_determinism,
)


@pytest.fixture
def restore_torch_and_env(monkeypatch):
    deterministic = torch.are_deterministic_algorithms_enabled()
    cudnn = torch.backends.cudnn.deterministic
    benchmark = torch.backends.cudnn.benchmark
    workspace = {CUBLAS_WORKSPACE_CONFIG_ENV: None}
    for key in workspace:
        workspace[key] = __import__("os").environ.get(key)
    yield monkeypatch
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = cudnn
    torch.backends.cudnn.benchmark = benchmark
    for key, value in workspace.items():
        if value is None:
            __import__("os").environ.pop(key, None)
        else:
            __import__("os").environ[key] = value


def _train_toy_run(seed):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    inputs = torch.randn(64, 16)
    targets = torch.arange(64) % 4
    for _ in range(8):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
    return [parameter.detach().clone() for parameter in model.parameters()]


def test_configure_determinism_pins_the_algorithms(restore_torch_and_env):
    report = configure_determinism(11)

    assert report.deterministic_algorithms is True
    assert report.reproducible is True
    assert report.cublas_workspace_config == CUBLAS_WORKSPACE_CONFIG
    assert report.warnings == []
    if torch.cuda.is_available():
        assert report.cudnn_deterministic is True
        assert report.cudnn_benchmark is False


def test_wrong_cublas_workspace_is_reported_instead_of_trusted(restore_torch_and_env):
    restore_torch_and_env.setenv(CUBLAS_WORKSPACE_CONFIG_ENV, ":16:8")

    report = configure_determinism(11)

    assert report.reproducible is False
    assert any(CUBLAS_WORKSPACE_CONFIG_ENV in warning for warning in report.warnings)


def test_seeded_run_is_bit_identical(restore_torch_and_env):
    configure_determinism(3)
    first = _train_toy_run(3)
    configure_determinism(3)
    second = _train_toy_run(3)

    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_different_seeds_still_diverge(restore_torch_and_env):
    configure_determinism(3)
    first = _train_toy_run(3)
    configure_determinism(4)
    second = _train_toy_run(4)

    assert not all(torch.equal(a, b) for a, b in zip(first, second))