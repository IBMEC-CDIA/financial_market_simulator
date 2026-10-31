"""Unit tests for the market data collection command-line script."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from generative_models.data.collect_market_data import (
    build_argument_parser,
    main,
)

FETCH_PATH = (
    "generative_models.data.collect_market_data.fetch_price_series"
)


FETCHED_ROW_COUNT = 300


def _make_fetched_prices() -> pd.DataFrame:
    """Build a price frame in the format returned by yfinance.

    The frame has enough rows and small daily moves to pass the quality
    checks of the collection step.
    """
    rng = np.random.default_rng(7)
    price_values = 100 * np.exp(
        np.cumsum(rng.normal(0.0, 0.01, FETCHED_ROW_COUNT))
    )
    index = pd.bdate_range(
        "2020-01-01",
        periods=FETCHED_ROW_COUNT,
        name="Date",
    )

    return pd.DataFrame({"close_price": price_values}, index=index)


def test_missing_output_path_raises_system_exit() -> None:
    """Check that argparse rejects a call without --output-path."""
    parser = build_argument_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--ticker", "AAPL"])


@patch(FETCH_PATH)
def test_main_saves_prices_with_log_returns(mock_fetch, tmp_path) -> None:
    """Check that the saved file has the prices and the log-returns."""
    mock_fetch.return_value = _make_fetched_prices()
    output_path = tmp_path / "AAPL.csv"

    returned_path = main(
        [
            "--ticker",
            "AAPL",
            "--output-path",
            str(output_path),
        ]
    )

    saved_data = pd.read_csv(output_path, index_col=0)
    assert returned_path == output_path
    assert list(saved_data.columns) == [
        "close_price",
        "log_return",
        "split",
    ]
    assert len(saved_data) == FETCHED_ROW_COUNT
    assert np.isnan(saved_data["log_return"].iloc[0])
    assert np.isnan(saved_data["split"].iloc[0])


@patch(FETCH_PATH)
def test_main_uses_default_dates_and_ticker(mock_fetch, tmp_path) -> None:
    """Check that the fetch uses the ticker and the default date range."""
    mock_fetch.return_value = _make_fetched_prices()

    main(
        [
            "--ticker",
            "PETR4.SA",
            "--output-path",
            str(tmp_path / "PETR4.SA.csv"),
        ]
    )

    fetch_arguments = mock_fetch.call_args.args
    assert fetch_arguments[0] == "PETR4.SA"
    assert fetch_arguments[1] == "2020-01-01"
    assert fetch_arguments[2] == "2026-09-15"
