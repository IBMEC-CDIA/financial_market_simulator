"""Market data collection utilities based on yfinance."""

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_price_series(
    ticker_symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch daily closing price data for a given ticker using yfinance.

    Parameters
    ----------
    ticker_symbol : str
        Ticker symbol of the asset to fetch (for example, `"AAPL"`).
    start_date : str
        Start date of the historical range, in `"YYYY-MM-DD"` format.
    end_date : str
        End date of the historical range, in `"YYYY-MM-DD"` format.

    Returns
    -------
    pd.DataFrame
        Single-column data frame indexed by date, with the daily
        closing price stored in the `"close_price"` column.

    Example
    -------
    >>> price_data = fetch_price_series(
    ...     ticker_symbol="AAPL",
    ...     start_date="2020-01-01",
    ...     end_date="2026-01-01",
    ... )
    >>> list(price_data.columns)
    ['close_price']
    """
    raw_data = yf.download(
        ticker_symbol,
        start=start_date,
        end=end_date,
        progress=True,
    )
    price_data = raw_data[["Close"]].dropna()
    price_data.columns = ["close_price"]

    return price_data


def compute_log_return_values(
    price_values: np.ndarray,
) -> np.ndarray:
    """Compute the daily log-return series from a price series.

    Parameters
    ----------
    price_values : np.ndarray
        One-dimensional array of consecutive price values.

    Returns
    -------
    np.ndarray
        One-dimensional array with one fewer element than
        `price_values`, where each entry is
        `log(price[t]) - log(price[t - 1])`.

    Example
    -------
    >>> compute_log_return_values(np.array([100.0, 110.0]))
    array([0.0953102], dtype=float32)
    """
    log_price_values = np.log(price_values)
    log_return_values = log_price_values[1:] - log_price_values[:-1]

    return log_return_values.astype(np.float32)
