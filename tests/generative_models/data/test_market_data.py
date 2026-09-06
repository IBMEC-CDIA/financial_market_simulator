"""Unit tests for market data collection utilities."""

import numpy as np

from generative_models.data.market_data import compute_log_return_values


def test_compute_log_return_values_matches_manual_calculation() -> None:
    """Check that the log-return formula matches a manual calculation."""
    price_values = np.array([100.0, 110.0, 99.0], dtype=np.float32)

    log_return_values = compute_log_return_values(price_values)

    expected_log_return_values = np.array(
        [
            np.log(110.0) - np.log(100.0),
            np.log(99.0) - np.log(110.0),
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        log_return_values, expected_log_return_values, rtol=1e-6
    )


def test_compute_log_return_values_has_one_fewer_element() -> None:
    """Check that the output series has one fewer element than the input."""
    price_values = np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float32)

    log_return_values = compute_log_return_values(price_values)

    assert len(log_return_values) == len(price_values) - 1
