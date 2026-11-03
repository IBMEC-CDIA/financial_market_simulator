"""Unit tests for the price quality checks."""

import numpy as np
import pandas as pd
import pytest

from generative_models.data.validation import (
    MINIMUM_ROW_COUNT,
    PriceValidationError,
    build_price_frame,
    build_price_expectations,
    validate_price_dataset,
)


def _make_valid_prices(number_of_prices: int = 300) -> pd.DataFrame:
    """Build a price series that passes every quality check."""
    rng = np.random.default_rng(3)
    price_values = 100 * np.exp(
        np.cumsum(rng.normal(0.0, 0.01, number_of_prices))
    )
    index = pd.bdate_range(
        "2020-01-01",
        periods=number_of_prices,
        name="Date",
    )

    return pd.DataFrame({"close_price": price_values}, index=index)


def test_valid_prices_pass_the_checks() -> None:
    """Check that a clean price series is accepted without an error."""
    validate_price_dataset(_make_valid_prices())


def test_negative_price_is_rejected() -> None:
    """Check that a non-positive closing price fails the checks."""
    prices = _make_valid_prices()
    prices.iloc[10, 0] = -5.0

    with pytest.raises(PriceValidationError, match="close_price"):
        validate_price_dataset(prices)


def test_duplicated_dates_are_rejected() -> None:
    """Check that a repeated date fails the uniqueness check."""
    prices = _make_valid_prices()
    duplicated = pd.concat([prices, prices.iloc[[5]]]).sort_index()

    with pytest.raises(PriceValidationError, match="date"):
        validate_price_dataset(duplicated)


def test_short_series_is_rejected() -> None:
    """Check that a series below the minimum row count fails the checks."""
    short_prices = _make_valid_prices(MINIMUM_ROW_COUNT - 1)

    with pytest.raises(PriceValidationError, match="row_count"):
        validate_price_dataset(short_prices)


def test_extreme_daily_move_is_rejected() -> None:
    """Check that a daily log-return above the limit fails the checks."""
    prices = _make_valid_prices()
    prices.iloc[100:, 0] *= 3.0

    with pytest.raises(PriceValidationError, match="log_return"):
        validate_price_dataset(prices)


def test_build_price_frame_adds_the_log_return_column() -> None:
    """Check that the frame has the date, close and log-return columns."""
    frame = build_price_frame(_make_valid_prices(5))

    assert list(frame.columns) == ["date", "close_price", "log_return"]
    assert np.isnan(frame["log_return"].iloc[0])


def test_build_price_expectations_lists_every_check() -> None:
    """Check that the suite has one expectation per quality rule."""
    assert len(build_price_expectations()) == 6
