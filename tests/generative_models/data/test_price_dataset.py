"""Unit tests for the model-ready price dataset builder."""

import numpy as np
import pandas as pd
import pytest

from generative_models.data.price_dataset import build_price_dataset
from generative_models.data.validation import PriceValidationError


def _make_prices(number_of_prices: int = 300) -> pd.DataFrame:
    """Build a steady price series that passes every quality check."""
    price_values = np.linspace(100.0, 130.0, number_of_prices)
    index = pd.date_range(
        "2020-01-01",
        periods=number_of_prices,
        freq="B",
        name="Date",
    )

    return pd.DataFrame({"close_price": price_values}, index=index)


def test_build_price_dataset_adds_returns_and_split_labels() -> None:
    """Check that the dataset has the return and split columns."""
    dataset = build_price_dataset(
        "AAPL",
        "2020-01-01",
        "2026-01-01",
        train_split_ratio=0.7,
        validation_split_ratio=0.15,
        fetch_function=lambda ticker, start, end: _make_prices(),
    )

    assert list(dataset.columns) == ["close_price", "log_return", "split"]
    assert set(dataset["split"].dropna()) == {"train", "validation", "test"}


def test_build_price_dataset_passes_the_dates_to_the_fetch_function() -> None:
    """Check that the ticker and the dates reach the fetch function."""
    received = []

    def fake_fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
        received.append((ticker, start, end))
        return _make_prices()

    build_price_dataset(
        "MSFT",
        "2021-01-01",
        "2026-01-01",
        train_split_ratio=0.7,
        validation_split_ratio=0.15,
        fetch_function=fake_fetch,
    )

    assert received == [("MSFT", "2021-01-01", "2026-01-01")]


def test_build_price_dataset_rejects_invalid_prices() -> None:
    """Check that prices that fail the quality checks are not accepted."""
    prices = _make_prices()
    prices.iloc[10, 0] = -1.0

    with pytest.raises(PriceValidationError):
        build_price_dataset(
            "AAPL",
            "2020-01-01",
            "2026-01-01",
            train_split_ratio=0.7,
            validation_split_ratio=0.15,
            fetch_function=lambda ticker, start, end: prices,
        )
