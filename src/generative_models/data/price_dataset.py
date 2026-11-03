"""Build the model-ready price dataset of a ticker from raw prices."""

from collections.abc import Callable

import pandas as pd

from generative_models.data.market_data import (
    add_log_return_column,
    add_split_column,
    fetch_price_series,
)
from generative_models.data.validation import validate_price_dataset


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def build_price_dataset(
    ticker: str,
    start_date: str,
    end_date: str,
    train_split_ratio: float,
    validation_split_ratio: float,
    fetch_function: Callable[[str, str, str], pd.DataFrame] = (
        fetch_price_series
    ),
) -> pd.DataFrame:
    """Download, validate and enrich the prices of a ticker.

    The raw prices are checked with the quality rules of
    `validate_price_dataset` before anything else is computed. Then the
    log-returns and the train, validation and test labels are added.

    Parameters
    ----------
    ticker : str
        Ticker symbol to download, for example `"AAPL"`.
    start_date : str
        First date of the historical range, in `"YYYY-MM-DD"` format.
    end_date : str
        Last date of the historical range, in `"YYYY-MM-DD"` format.
    train_split_ratio : float
        Fraction of the log-returns marked as `"train"`.
    validation_split_ratio : float
        Fraction of the log-returns marked as `"validation"`.
    fetch_function : Callable[[str, str, str], pd.DataFrame]
        Function that downloads the raw prices, called with the ticker,
        the start date and the end date. Defaults to `fetch_price_series`.

    Returns
    -------
    pd.DataFrame
        Data frame indexed by date, with the `close_price`, `log_return`
        and `split` columns.

    Raises
    ------
    PriceValidationError
        If the raw prices fail the quality checks.

    Example
    -------
    >>> dataset = build_price_dataset(
    ...     "AAPL",
    ...     "2020-01-01",
    ...     "2026-09-15",
    ...     train_split_ratio=0.7,
    ...     validation_split_ratio=0.15,
    ... )
    >>> list(dataset.columns)
    ['close_price', 'log_return', 'split']
    """
    raw_prices = fetch_function(ticker, start_date, end_date)
    validate_price_dataset(raw_prices)

    return add_split_column(
        add_log_return_column(raw_prices),
        train_split_ratio,
        validation_split_ratio,
    )
