"""Base interface for registries of models kept in memory."""

# pylint: disable=duplicate-code

from typing import Any


class ModelRegistry:
    """Interface of a registry that keeps models in memory by ticker.

    A registry reads trained models from some source (for example,
    Weights & Biases or the local disk), keeps them in memory and
    returns the model that belongs to a given ticker. This class only
    declares the methods every registry must provide; subclasses
    implement how the models are read and stored.

    Example
    -------
    >>> class DictModelRegistry(ModelRegistry):
    ...     def get_model(self, ticker):
    ...         return self.models[ticker]
    """

    def __init__(self) -> None:
        self.models: dict[str, Any] = {}

    def load_model(self, ticker: str) -> Any:
        """Read the model of a ticker and keep it in memory.

        Parameters
        ----------
        ticker : str
            Symbol of the model that must be read from the source.

        Returns
        -------
        Any
            The model read from the source, now kept in memory.
        """
        raise NotImplementedError

    def load_all_models(self) -> list[str]:
        """Read every available model and keep them in memory.

        Returns
        -------
        list[str]
            Ticker symbols of the loaded models.
        """
        raise NotImplementedError

    def get_model(self, ticker: str) -> Any:
        """Return the in-memory model selected by a ticker.

        Parameters
        ----------
        ticker : str
            Symbol used to choose which model is returned.

        Returns
        -------
        Any
            The model previously loaded for that ticker.
        """
        raise NotImplementedError

    def list_loaded_tickers(self) -> list[str]:
        """List the tickers whose models are currently in memory.

        Returns
        -------
        list[str]
            Ticker symbols of the models in memory.
        """
        raise NotImplementedError
