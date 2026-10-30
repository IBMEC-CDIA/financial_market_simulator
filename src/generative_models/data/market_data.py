"""Market data collection utilities based on yfinance."""

from pathlib import Path

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


def add_log_return_column(
    price_data: pd.DataFrame,
) -> pd.DataFrame:
    """Add the daily log-return of each price as a `"log_return"` column.

    The first row has no previous price, so its log-return is `NaN`. The
    rows are kept, so the closing prices stay complete.

    Parameters
    ----------
    price_data : pd.DataFrame
        Data frame indexed by date, with a `"close_price"` column.

    Returns
    -------
    pd.DataFrame
        Copy of `price_data` with the `"log_return"` column added.

    Example
    -------
    >>> dataset = add_log_return_column(price_data)
    >>> list(dataset.columns)
    ['close_price', 'log_return']
    """
    dataset = price_data.copy()
    dataset["log_return"] = np.log(dataset["close_price"]).diff()

    return dataset


def add_split_column(
    dataset: pd.DataFrame,
    train_split_ratio: float,
) -> pd.DataFrame:
    """Add a `"split"` column marking each log-return as train or validation.

    The log-returns are split in chronological order: the first
    `train_split_ratio` share is `"train"` and the rest is `"validation"`.
    Rows without a log-return, such as the first row, stay empty.

    Parameters
    ----------
    dataset : pd.DataFrame
        Data frame with a `"log_return"` column.
    train_split_ratio : float
        Fraction of the log-returns, in chronological order, marked as
        `"train"`. It must be between 0 and 1, exclusive.

    Returns
    -------
    pd.DataFrame
        Copy of `dataset` with the `"split"` column added.

    Raises
    ------
    ValueError
        If `train_split_ratio` is not between 0 and 1, exclusive.

    Example
    -------
    >>> dataset = add_split_column(dataset, train_split_ratio=0.8)
    >>> dataset["split"].dropna().value_counts()["train"]
    4
    """
    if not 0 < train_split_ratio < 1:
        raise ValueError("train_split_ratio must be between 0 and 1.")

    labeled_dataset = dataset.copy()
    has_return = labeled_dataset["log_return"].notna()
    return_count = int(has_return.sum())
    train_count = int(return_count * train_split_ratio)
    split_labels = np.where(
        np.arange(return_count) < train_count,
        "train",
        "validation",
    )
    labeled_dataset["split"] = pd.Series(
        None,
        index=labeled_dataset.index,
        dtype=object,
    )
    labeled_dataset.loc[has_return, "split"] = split_labels

    return labeled_dataset


def save_price_series(
    price_data: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save a price series to a CSV file, creating its directory if needed.

    Parameters
    ----------
    price_data : pd.DataFrame
        Data frame indexed by date, with a `"close_price"` column.
    output_path : str | Path
        Path of the CSV file to write.

    Returns
    -------
    Path
        Path of the written file.

    Example
    -------
    >>> save_price_series(price_data, "artifacts/market-data/AAPL.csv")
    PosixPath('artifacts/market-data/AAPL.csv')
    """
    file_path = Path(output_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    price_data.to_csv(file_path, index_label="date")

    return file_path


def load_price_series(
    input_path: str | Path,
) -> pd.DataFrame:
    """Load a price series saved by `save_price_series`.

    Parameters
    ----------
    input_path : str | Path
        Path of the CSV file to read.

    Returns
    -------
    pd.DataFrame
        Data frame indexed by date, with the `"close_price"` column.

    Raises
    ------
    ValueError
        If the file has no `"close_price"` column.

    Example
    -------
    >>> price_data = load_price_series("artifacts/market-data/AAPL.csv")
    >>> list(price_data.columns)
    ['close_price']
    """
    price_data = pd.read_csv(input_path, index_col=0, parse_dates=True)
    if "close_price" not in price_data.columns:
        raise ValueError(
            f"The file '{input_path}' has no 'close_price' column."
        )

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
