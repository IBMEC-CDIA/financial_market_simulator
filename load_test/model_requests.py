"""Helpers that prepare the requests sent to the autoencoder API."""

from datetime import date, timedelta
from typing import Any


def tickers_from_payload(payload: Any) -> list[str]:
    """Extract the unique tickers of the models reported by the API.

    Parameters
    ----------
    payload : Any
        JSON body of `GET /models/loaded`, a list of model metadata.

    Returns
    -------
    list[str]
        Sorted unique tickers in upper case. Entries without a ticker are
        ignored.

    Raises
    ------
    ValueError
        If the payload is not a list.

    Example
    -------
    >>> tickers_from_payload([{"ticker": "aapl"}, {"ticker": "AAPL"}])
    ['AAPL']
    """
    if not isinstance(payload, list):
        raise ValueError("The models payload must be a list.")
    tickers = {
        str(item["ticker"]).strip().upper()
        for item in payload
        if isinstance(item, dict) and item.get("ticker")
    }
    return sorted(tickers)


def history_period(today: date, days: int) -> tuple[str, str]:
    """Return the start and end dates of a window ending today.

    Parameters
    ----------
    today : date
        Last day of the window.
    days : int
        Number of days the window goes back from `today`.

    Returns
    -------
    tuple[str, str]
        Start and end dates in `YYYY-MM-DD` format.

    Example
    -------
    >>> history_period(date(2026, 11, 10), 90)
    ('2026-08-12', '2026-11-10')
    """
    start = today - timedelta(days=days)
    return start.isoformat(), today.isoformat()
