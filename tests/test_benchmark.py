import numpy as np
import pytest

from my_jev.benchmark import (
    _kl,
    _percentile,
    _uniform_like,
)


def test_uniform_baseline_matches_option_count():
    probs = _uniform_like(
        np.array([0.7, 0.2, 0.1])
    )
    assert probs.tolist() == pytest.approx(
        [1 / 3, 1 / 3, 1 / 3]
    )


def test_state_sensitivity_kl_is_zero_for_identical_distributions():
    probs = np.array([0.8, 0.2])
    assert _kl(probs, probs) == pytest.approx(0.0)


def test_percentile_interpolates():
    assert _percentile(
        [10.0, 20.0, 30.0],
        0.5,
    ) == pytest.approx(20.0)
