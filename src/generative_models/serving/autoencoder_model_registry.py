"""In-memory registry of autoencoders read from Weights & Biases."""

import tempfile
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv
from wandb.errors import CommError

from generative_models.models.autoencoder.simple_autoencoder import (
    SimpleAutoencoder,
)
from generative_models.serving.model_registry import ModelRegistry


class AutoencoderModelRegistry(ModelRegistry):
    """Read trained autoencoders from wandb and keep them in memory.

    Each model is a wandb artifact of type `"model"`, named after its
    ticker (by default `simple-autoencoder-<ticker>`, the name used
    when the model is logged during training), that contains the
    `state_dict` of a `SimpleAutoencoder`. The artifact is downloaded
    to a temporary directory, loaded and discarded from the disk; the
    model itself stays in memory, in evaluation mode, and can be
    retrieved by its ticker without contacting wandb again. The
    architecture (window size, hidden size and latent size) is
    inferred from the shapes of the stored weights.

    The wandb API key is read from the `WANDB_API_KEY` environment
    variable, which can be provided through a local `.env` file.

    Parameters
    ----------
    project_name : str
        Name of the wandb project where the models were logged.
    entity : str, optional
        wandb user or team that owns the project. When `None`, the
        default entity of the authenticated user is used.
    artifact_name_template : str
        Template of the artifact names, with a `{ticker}` placeholder
        replaced by the lower-case ticker symbol.
    artifact_alias : str
        Version or alias of the artifacts to read (for example,
        `"latest"` or `"v0"`).
    device : str
        Torch device where the models are kept (for example, `"cpu"`
        or `"cuda"`).

    Example
    -------
    >>> registry = AutoencoderModelRegistry(
    ...     project_name="autoencoder-ticker-models",
    ... )
    >>> registry.load_all_models()
    ['AAPL', 'GOOGL', 'MSFT', 'NVDA']
    >>> autoencoder = registry.get_model("aapl")
    """

    def __init__(
        self,
        project_name: str = "autoencoder-ticker-models",
        entity: str | None = None,
        artifact_name_template: str = "simple-autoencoder-{ticker}",
        artifact_alias: str = "latest",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        load_dotenv()

        self.project_name = project_name
        self.entity = entity
        self.artifact_name_template = artifact_name_template
        self.artifact_alias = artifact_alias
        self.device = device
        self._api = None

    def load_model(self, ticker: str) -> SimpleAutoencoder:
        """Download the model artifact of a ticker and keep it in memory.

        If the ticker was already loaded, the artifact is downloaded
        again and the model in memory is replaced.

        Parameters
        ----------
        ticker : str
            Ticker symbol of the model to load (case-insensitive).

        Returns
        -------
        SimpleAutoencoder
            The loaded autoencoder, in evaluation mode.

        Raises
        ------
        LookupError
            If wandb has no model artifact for `ticker` in the project.
        """
        normalized_ticker = self._normalize_ticker(ticker)
        artifact_path = self._build_artifact_path(normalized_ticker)

        try:
            artifact = self._get_api().artifact(artifact_path)
        except CommError as communication_error:
            raise LookupError(
                f"No model artifact for ticker '{normalized_ticker}' "
                f"at '{artifact_path}'."
            ) from communication_error

        with tempfile.TemporaryDirectory() as download_directory:
            artifact.download(root=download_directory)
            model_file_path = next(Path(download_directory).rglob("*.pt"))
            state_dict = torch.load(
                model_file_path,
                map_location=self.device,
                weights_only=True,
            )

        autoencoder = SimpleAutoencoder(
            input_dim=state_dict["encoder.0.weight"].shape[1],
            hidden_dim=state_dict["encoder.0.weight"].shape[0],
            latent_dim=state_dict["encoder.2.weight"].shape[0],
        )
        autoencoder.load_state_dict(state_dict)
        autoencoder.to(self.device)
        autoencoder.eval()

        self.models[normalized_ticker] = autoencoder
        return autoencoder

    def load_all_models(self) -> list[str]:
        """Load every model artifact found in the wandb project.

        Returns
        -------
        list[str]
            Ticker symbols of the loaded models, in alphabetical
            order.
        """
        available_tickers = self._list_available_tickers()
        for ticker in available_tickers:
            self.load_model(ticker)
        return available_tickers

    def get_model(self, ticker: str) -> SimpleAutoencoder:
        """Get the in-memory model that belongs to a ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol of the model to retrieve (case-insensitive).

        Returns
        -------
        SimpleAutoencoder
            The autoencoder loaded for `ticker`.

        Raises
        ------
        KeyError
            If no model was loaded for `ticker`.
        """
        normalized_ticker = self._normalize_ticker(ticker)
        if normalized_ticker not in self.models:
            raise KeyError(
                f"No model loaded for ticker '{normalized_ticker}'. "
                "Call load_model or load_all_models first."
            )
        return self.models[normalized_ticker]

    def list_loaded_tickers(self) -> list[str]:
        """List the tickers whose models are currently in memory.

        Returns
        -------
        list[str]
            Ticker symbols, in the order the models were loaded.
        """
        return list(self.models)

    def _get_api(self) -> wandb.Api:
        """Create the wandb API client on first use and reuse it."""
        if self._api is None:
            self._api = wandb.Api()
        return self._api

    def _get_entity(self) -> str:
        """Return the configured entity or the user's default one."""
        return self.entity or self._get_api().default_entity

    def _build_artifact_path(self, ticker: str) -> str:
        """Build the full wandb path of the artifact of a ticker."""
        artifact_name = self.artifact_name_template.format(
            ticker=ticker.lower()
        )
        return (
            f"{self._get_entity()}/{self.project_name}/"
            f"{artifact_name}:{self.artifact_alias}"
        )

    def _list_available_tickers(self) -> list[str]:
        """List the tickers that have a model artifact in the project."""
        prefix, suffix = self.artifact_name_template.split("{ticker}")
        project_path = f"{self._get_entity()}/{self.project_name}"
        collections = self._get_api().artifact_type(
            "model", project_path
        ).collections()

        collection_names = [collection.name for collection in collections]

        return sorted(
            self._normalize_ticker(name[len(prefix):len(name) - len(suffix)])
            for name in collection_names
            if name.startswith(prefix) and name.endswith(suffix)
        )

    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        """Return the ticker in upper case, without surrounding spaces."""
        return ticker.strip().upper()
