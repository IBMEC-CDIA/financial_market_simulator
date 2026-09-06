"""Unit tests for normalize_series."""

import torch

from generative_models.common.normalization import normalize_series


def test_normalize_series_returns_zero_mean_unit_std() -> None:
    """Check that the standardized series has mean 0 and std 1."""
    series_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    normalized_series_values, series_mean, series_std = normalize_series(
        series_values
    )

    assert series_mean == series_values.mean().item()
    assert series_std == series_values.std().item()
    assert torch.isclose(
        normalized_series_values.mean(), torch.tensor(0.0), atol=1e-6
    )
    assert torch.isclose(
        normalized_series_values.std(), torch.tensor(1.0), atol=1e-6
    )


def test_normalize_series_applies_given_statistics_correctly() -> None:
    """Check the standardization formula against manually computed values."""
    series_values = torch.tensor([0.0, 10.0])

    normalized_series_values, series_mean, series_std = normalize_series(
        series_values
    )

    expected_normalized_values = (series_values - series_mean) / series_std
    assert torch.equal(normalized_series_values, expected_normalized_values)
