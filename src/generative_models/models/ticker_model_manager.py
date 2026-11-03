"""Management of models trained for a set of ticker symbols."""

from collections.abc import Callable
from typing import Any

import pandas as pd

from generative_models.data.price_dataset import build_price_dataset
from generative_models.models.base_autoencoder.config import MarketDataConfig
from generative_models.models.base_autoencoder.trainer_factory import (
    build_trainer,
)

TrainerFactory = Callable[[str, pd.DataFrame | None], Any]


class TickerModelManager:
    """Manage models trained for a set of ticker symbols.

    Each ticker is associated with the date range (`start_date` and
    `end_date`) used to collect its historical data. The manager downloads
    and prepares the prices of each ticker, creates its trainer with
    `trainer_factory`, runs the training and keeps the result.

    Parameters
    ----------
    tickers : dict[str, dict[str, str]]
        Dictionary mapping each ticker symbol to a dictionary with its
        `start_date` and `end_date`, both in the format
        `"YYYY-MM-DD"`.
    trainer_factory : Callable[[str, pd.DataFrame | None], Any], optional
        Function that creates the trainer of a ticker, called with the
        ticker and its prepared prices. When `None`, the trainer of the
        base autoencoder is used. See `build_trainer`.

    Attributes
    ----------
    tickers : dict[str, dict[str, str]]
        Configuration of each ticker.
    trainer_factory : Callable[[str, pd.DataFrame | None], Any]
        Function that creates the trainer of a ticker.
    models : dict[str, Any]
        Result of the training of each ticker that has been trained.

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
    >>> manager.train_all_models()
    >>> manager.get_model("AAPL").anomaly_threshold
    0.42
    """

    def __init__(
        self,
        tickers: dict[str, dict[str, str]],
        trainer_factory: TrainerFactory | None = None,
    ) -> None:
        """Store the tickers and the trainer factory. No model is trained."""
        self.tickers = tickers
        self.trainer_factory = trainer_factory or build_trainer
        self.models: dict[str, Any] = {}

    def list_tickers(self) -> list[str]:
        """List every ticker symbol managed by this instance.

        Returns
        -------
        list[str]
            Ticker symbols, in the same order they were provided in
            the constructor.
        """
        return list(self.tickers.keys())

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

        Raises
        ------
        KeyError
            If `ticker` is not managed by this instance.
        """
        ticker_configuration = self.tickers[ticker]

        return (
            ticker_configuration["start_date"],
            ticker_configuration["end_date"],
        )

    def fetch_price_data(self, ticker: str) -> pd.DataFrame:
        """Fetch the historical price series for a ticker.

        The prices are downloaded for the date range of the ticker, checked
        with the quality rules, and enriched with the log-returns and the
        train, validation and test labels. The result is the same data the
        `collect-data` job saves.

        Parameters
        ----------
        ticker : str
            Ticker symbol whose price series should be fetched, using
            the date range configured in the constructor.

        Returns
        -------
        pandas.DataFrame
            Historical price series for `ticker`, with the `close_price`,
            `log_return` and `split` columns.

        Raises
        ------
        KeyError
            If `ticker` is not managed by this instance.
        PriceValidationError
            If the downloaded prices fail the quality checks.
        """
        start_date, end_date = self.get_date_range(ticker)
        market_data_config = MarketDataConfig()

        return build_price_dataset(
            ticker,
            start_date,
            end_date,
            market_data_config.train_split_ratio,
            market_data_config.validation_split_ratio,
        )

    def train_model(self, ticker: str) -> Any:
        """Train and register the model for a ticker.

        The prices of the ticker are fetched, the trainer is created with
        `trainer_factory`, and its `train` method is run. The result is kept
        in `models`, replacing any earlier result of the same ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol whose model should be trained. The trained
            model is stored internally and can later be retrieved
            with `get_model`.

        Returns
        -------
        Any
            Result returned by the trainer of the ticker.
        """
        price_data = self.fetch_price_data(ticker)
        trainer = self.trainer_factory(ticker, price_data)
        result = trainer.train()
        self.models[ticker] = result

        return result

    def train_all_models(self) -> None:
        """Train and register the model of every managed ticker.

        The tickers are trained sequentially, one after the other, in
        the order returned by `list_tickers`, by calling `train_model`
        for each of them. An error stops the loop, and the tickers after
        it are not trained.
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
            The result of the training of `ticker`.

        Raises
        ------
        KeyError
            If the ticker has not been trained yet.
        """
        if ticker not in self.models:
            raise KeyError(
                f"No model trained for ticker '{ticker}'. "
                "Call train_model first."
            )

        return self.models[ticker]
