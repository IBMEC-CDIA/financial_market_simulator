"""Registry that loads trained autoencoders from wandb artifacts."""

from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv

from generative_models.models.autoencoder.autoencoder import SimpleAutoencoder
from generative_models.models.autoencoder.config import TrackingConfig
from generative_models.models.autoencoder.trainer import (
    MODEL_NAME_PREFIX,
    build_ticker_slug,
)
from generative_models.serving.autoencoder_predictor import (
    AutoencoderModelMetadata,
    AutoencoderPredictor,
)
from generative_models.serving.model_registry import ModelRegistry

REQUIRED_METADATA_KEYS = (
    "ticker",
    "window_size",
    "hidden_layer_size",
    "latent_layer_size",
    "return_train_mean",
    "return_train_std",
    "anomaly_threshold",
)


class WandbAutoencoderRegistry(ModelRegistry):
    """Load the autoencoders registered by AutoencoderTrainer in wandb.

    The registry lists the model artifacts of a wandb project,
    downloads the weights, rebuilds each `SimpleAutoencoder` with the
    architecture saved in the artifact metadata and keeps an
    `AutoencoderPredictor` in memory per ticker. The wandb client is
    only created the first time it is needed.

    Parameters
    ----------
    project_name : str
        Name of the wandb project where the models were registered.
    entity : str | None
        wandb entity (user or team) that owns the project. `None`
        uses the default entity of the API key.
    download_directory : str
        Local directory where the artifacts are downloaded.
    env_file_path : str
        Path to the `.env` file containing `WANDB_API_KEY`.
    device : str | None
        Torch device of the loaded models. `None` picks `"cuda"` when
        available and `"cpu"` otherwise.

    Example
    -------
    >>> registry = WandbAutoencoderRegistry(entity="my-team")
    >>> predictor = registry.load_model("PETR4.SA")
    >>> registry.list_loaded_tickers()
    ['PETR4.SA']
    """

    def __init__(
        self,
        project_name: str = TrackingConfig.project_name,
        entity: str | None = None,
        download_directory: str = "artifacts/wandb",
        env_file_path: str = ".env",
        device: str | None = None,
    ) -> None:
        """Store the settings without accessing wandb."""
        super().__init__()
        self.project_name = project_name
        self.entity = entity
        self.download_directory = Path(download_directory)
        self.env_file_path = env_file_path
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._api = None

    @property
    def api(self) -> wandb.Api:
        """Return the wandb client, creating it on the first access."""
        if self._api is None:
            load_dotenv(self.env_file_path)
            self._api = wandb.Api()
        return self._api

    @property
    def project_path(self) -> str:
        """Return the `"<entity>/<project_name>"` path of the project."""
        entity = self.entity
        if entity is None:
            entity = self.api.default_entity
        return f"{entity}/{self.project_name}"

    def list_available_artifacts(self) -> list[wandb.Artifact]:
        """List the latest servable autoencoder artifact of each ticker.

        Returns
        -------
        list[wandb.Artifact]
            Latest version of every model collection whose name starts
            with `MODEL_NAME_PREFIX` and whose metadata has every key
            of `REQUIRED_METADATA_KEYS`.
        """
        project_path = self.project_path
        artifact_collections = self.api.artifact_collections(
            project_name=project_path,
            type_name="model",
        )
        available_artifacts = []
        for artifact_collection in artifact_collections:
            if not artifact_collection.name.startswith(MODEL_NAME_PREFIX):
                continue
            artifact = self.api.artifact(
                f"{project_path}/{artifact_collection.name}:latest"
            )
            if self._has_required_metadata(artifact):
                available_artifacts.append(artifact)
        return available_artifacts

    def load_latest_model(self) -> AutoencoderPredictor:
        """Load the most recently created servable artifact.

        Returns
        -------
        AutoencoderPredictor
            Predictor of the artifact with the greatest `created_at`.

        Raises
        ------
        LookupError
            If the project has no servable autoencoder artifact.
        """
        available_artifacts = self.list_available_artifacts()
        if not available_artifacts:
            raise LookupError(
                "No servable autoencoder artifact found in "
                f"'{self.project_path}'."
            )
        latest_artifact = max(
            available_artifacts,
            key=lambda artifact: artifact.created_at,
        )
        return self._load_artifact(latest_artifact)

    def load_model(self, ticker: str) -> AutoencoderPredictor:
        """Load the latest artifact of a ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol of the model to load.

        Returns
        -------
        AutoencoderPredictor
            Predictor of the latest artifact of `ticker`.

        Raises
        ------
        ValueError
            If the artifact metadata misses a required key.
        """
        artifact_path = (
            f"{self.project_path}/{MODEL_NAME_PREFIX}"
            f"{build_ticker_slug(ticker)}:latest"
        )
        artifact = self.api.artifact(artifact_path, type="model")
        if not self._has_required_metadata(artifact):
            raise ValueError(
                f"Artifact '{artifact_path}' is missing metadata required "
                f"for inference: {', '.join(REQUIRED_METADATA_KEYS)}."
            )
        return self._load_artifact(artifact)

    def load_all_models(self) -> list[str]:
        """Load every servable artifact of the project.

        Returns
        -------
        list[str]
            Ticker symbols of the loaded models.
        """
        return [
            self._load_artifact(artifact).metadata.ticker
            for artifact in self.list_available_artifacts()
        ]

    def get_model(self, ticker: str) -> AutoencoderPredictor:
        """Get the in-memory predictor of a ticker without using wandb.

        Parameters
        ----------
        ticker : str
            Ticker symbol of the model to retrieve.

        Returns
        -------
        AutoencoderPredictor
            Predictor loaded for `ticker`.

        Raises
        ------
        KeyError
            If no model was loaded for `ticker`.
        """
        if ticker not in self.models:
            raise KeyError(
                f"No model loaded for ticker '{ticker}'. "
                "Call load_model first."
            )
        return self.models[ticker]

    def list_loaded_tickers(self) -> list[str]:
        """List the tickers in memory, in loading order.

        Returns
        -------
        list[str]
            Ticker symbols of the loaded models.
        """
        return list(self.models)

    @staticmethod
    def _has_required_metadata(artifact: wandb.Artifact) -> bool:
        """Check whether an artifact has every required metadata key.

        Parameters
        ----------
        artifact : wandb.Artifact
            Artifact to check.

        Returns
        -------
        bool
            `True` when every key of `REQUIRED_METADATA_KEYS` is in the
            artifact metadata.
        """
        artifact_metadata = artifact.metadata or {}
        return all(key in artifact_metadata for key in REQUIRED_METADATA_KEYS)

    def _load_artifact(
        self,
        artifact: wandb.Artifact,
    ) -> AutoencoderPredictor:
        """Download an artifact and build its predictor.

        Parameters
        ----------
        artifact : wandb.Artifact
            Servable autoencoder artifact.

        Returns
        -------
        AutoencoderPredictor
            Predictor stored in `models` under the artifact ticker,
            replacing any previous predictor of the same ticker.

        Raises
        ------
        FileNotFoundError
            If the downloaded artifact has no `.pt` file.
        """
        collection_name = artifact.name.split(":")[0]
        versioned_name = f"{collection_name}:{artifact.version}"
        artifact_directory = self.download_directory / versioned_name.replace(
            ":",
            "-",
        )
        artifact.download(root=str(artifact_directory))

        weight_file_paths = sorted(artifact_directory.glob("*.pt"))
        if not weight_file_paths:
            raise FileNotFoundError(
                f"Artifact '{versioned_name}' has no .pt weights file."
            )

        artifact_metadata = artifact.metadata
        window_size = int(artifact_metadata["window_size"])
        model = SimpleAutoencoder(
            input_dim=window_size,
            hidden_dim=int(artifact_metadata["hidden_layer_size"]),
            latent_dim=int(artifact_metadata["latent_layer_size"]),
        )
        model.load_state_dict(
            torch.load(
                weight_file_paths[0],
                map_location=self.device,
                weights_only=True,
            )
        )
        model_metadata = AutoencoderModelMetadata(
            ticker=str(artifact_metadata["ticker"]),
            window_size=window_size,
            return_train_mean=float(artifact_metadata["return_train_mean"]),
            return_train_std=float(artifact_metadata["return_train_std"]),
            anomaly_threshold=float(artifact_metadata["anomaly_threshold"]),
            artifact_name=versioned_name,
        )
        predictor = AutoencoderPredictor(
            model=model.to(self.device),
            metadata=model_metadata,
            device=self.device,
        )
        self.models[model_metadata.ticker] = predictor
        return predictor
