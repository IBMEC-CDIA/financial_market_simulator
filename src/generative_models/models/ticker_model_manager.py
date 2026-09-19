"""Management of models trained for a set of ticker symbols."""

from typing import Any


class TickerModelManager:
    """Manage models trained for a set of ticker symbols.

    Each ticker is associated with the date range (`start_date` and
    `end_date`) used to collect its historical data, so that fetching,
    training and evaluation can later be driven consistently per
    ticker.

    Parameters
    ----------
    tickers : dict[str, dict[str, str]]
        Dictionary mapping each ticker symbol to a dictionary with its
        `start_date` and `end_date`, both in the format
        `"YYYY-MM-DD"`.

    Example
    -------
    >>> manager = TickerModelManager(
    ...     tickers={
    ...         "AAPL": {
    ...             "start_date": "2020-01-01",
    ...             "end_date": "2026-01-01",
    ...         },
    ...         "MSFT": {
    ...             "start_date": "2020-01-01",
    ...             "end_date": "2026-01-01",
    ...         },
    ...     },
    ... )
    """

    def __init__(
        self,
        tickers: dict[str, dict[str, str]],
    ) -> None:
        self.tickers = tickers
        self.models: dict[str, Any] = {}

    def list_tickers(self) -> list[str]:
        """List every ticker symbol managed by this instance.

        Returns
        -------
        list[str]
            Ticker symbols, in the same order they were provided in
            the constructor.
        """
        raise NotImplementedError

    def get_date_range(self, ticker: str) -> tuple[str, str]:
        """Get the configured date range for a ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol whose date range should be returned.

        Returns
        -------
        tuple[str, str]
            The `start_date` and `end_date` configured for `ticker`.
        """
        raise NotImplementedError

    def fetch_price_data(self, ticker: str):
        """Fetch the historical price series for a ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol whose price series should be fetched, using
            the date range configured in the constructor.

        Returns
        -------
        pandas.DataFrame
            Historical price series for `ticker`.
        """
        raise NotImplementedError

    def train_model(self, ticker: str) -> None:
        """Train and register the model for a ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol whose model should be trained. The trained
            model is stored internally and can later be retrieved
            with `get_model`.
        """
        raise NotImplementedError

    def train_all_models(self) -> None:
        """Train and register the model of every managed ticker.

        The tickers are trained sequentially, one after the other, in
        the order returned by `list_tickers`, by calling `train_model`
        for each of them.
        """
        for ticker in self.list_tickers():
            self.train_model(ticker)

    def get_model(self, ticker: str) -> Any:
        """Get the trained model registered for a ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol whose trained model should be returned.

        Returns
        -------
        Any
            The model trained for `ticker`.
        """
        raise NotImplementedError
