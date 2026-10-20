"""Registry that loads trained autoencoders from Weights & Biases."""

from pathlib import Path

import torch
from dotenv import load_dotenv
import wandb

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
    """Load `SimpleAutoencoder` model artifacts logged by the trainer.

    Artifacts are read from the wandb project used by
    `AutoencoderTrainer`, one artifact collection per ticker
    (`simple-autoencoder-<ticker>`). Only artifacts whose metadata
    holds every key in `REQUIRED_METADATA_KEYS` are considered, since
    the normalization statistics and the anomaly threshold are needed
    to run the model on new data. Loaded models are kept in memory as
    `AutoencoderPredictor` instances, keyed by ticker.

    The wandb API key is read from the `WANDB_API_KEY` variable of the
    `.env` file at `env_file_path`.

    Parameters
    ----------
    project_name : str
        wandb project holding the model artifacts. Defaults to the
        project used by `AutoencoderTrainer`.
    entity : str | None
        wandb entity (user or team) that owns the project. When
        `None`, the default entity of the API key is used.
    download_directory : str
        Local directory where artifact files are downloaded.
    env_file_path : str
        Path to the `.env` file containing `WANDB_API_KEY`.
    device : str | None
        Torch device the models are loaded on. When `None`, CUDA is
        used if available.

    Example
    -------
    >>> registry = WandbAutoencoderRegistry()
    >>> predictor = registry.load_latest_model()
    >>> predictor.metadata.ticker
    'PETR4.SA'
    >>> registry.load_model("AAPL")
    >>> registry.list_loaded_tickers()
    ['PETR4.SA', 'AAPL']
    """

    def __init__(
        self,
        project_name: str = TrackingConfig.project_name,
        entity: str | None = None,
        download_directory: str = "artifacts/wandb",
        env_file_path: str = ".env",
        device: str | None = None,
    ) -> None:
        """Store the wandb settings; the API client is created lazily."""
        super().__init__()
        self.project_name = project_name
        self.entity = entity
        self.download_directory = Path(download_directory)
        self.env_file_path = env_file_path
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._api = None

    @property
    def api(self) -> wandb.Api:
        """wandb public API client, created on first access.

        Returns
        -------
        wandb.Api
            Client authenticated with the key from the `.env` file.
        """
        if self._api is None:
            load_dotenv(dotenv_path=self.env_file_path)
            self._api = wandb.Api()
        return self._api

    @property
    def project_path(self) -> str:
        """Full wandb project path.

        Returns
        -------
        str
            Path in the form `<entity>/<project_name>`.
        """
        entity = self.entity or self.api.default_entity
        return f"{entity}/{self.project_name}"

    def list_available_artifacts(self) -> list[wandb.Artifact]:
        """List the latest servable artifact of each ticker.

        Returns
        -------
        list[wandb.Artifact]
            Latest version of every autoencoder collection in the
            project whose metadata holds all required keys.
        """
        available_artifacts = []
        for artifact_collection in self.api.artifact_collections(
            project_name=self.project_path,
            type_name="model",
        ):
            if not artifact_collection.name.startswith(MODEL_NAME_PREFIX):
                continue
            artifact = self.api.artifact(
                f"{self.project_path}/{artifact_collection.name}:latest"
            )
            if _has_required_metadata(artifact):
                available_artifacts.append(artifact)
        return available_artifacts

    def load_latest_model(self) -> AutoencoderPredictor:
        """Load the most recently trained autoencoder of any ticker.

        Returns
        -------
        AutoencoderPredictor
            Predictor built from the newest servable artifact.

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
        """Load the latest autoencoder trained for a ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol of the model to load (for example,
            `"PETR4.SA"`).

        Returns
        -------
        AutoencoderPredictor
            Predictor built from the `latest` artifact version.

        Raises
        ------
        ValueError
            If the artifact lacks any required metadata key.
        """
        artifact = self.api.artifact(
            f"{self.project_path}/{MODEL_NAME_PREFIX}"
            f"{build_ticker_slug(ticker)}:latest",
            type="model",
        )
        if not _has_required_metadata(artifact):
            raise ValueError(
                f"Artifact '{artifact.name}' is missing metadata required "
                f"for inference: {', '.join(REQUIRED_METADATA_KEYS)}."
            )
        return self._load_artifact(artifact)

    def load_all_models(self) -> list[str]:
        """Load the latest autoencoder of every ticker in the project.

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
        """Get the in-memory predictor of a ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol of a previously loaded model.

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
                f"No model loaded for ticker '{ticker}'. Call "
                "load_model first."
            )
        return self.models[ticker]

    def list_loaded_tickers(self) -> list[str]:
        """List the tickers whose models are in memory.

        Returns
        -------
        list[str]
            Ticker symbols, in loading order.
        """
        return list(self.models)

    def _load_artifact(
        self,
        artifact: wandb.Artifact,
    ) -> AutoencoderPredictor:
        """Download an artifact and rebuild its predictor.

        Parameters
        ----------
        artifact : wandb.Artifact
            Model artifact logged by `AutoencoderTrainer`.

        Returns
        -------
        AutoencoderPredictor
            Predictor with the artifact weights and metadata, also
            stored in `models` under its ticker.

        Raises
        ------
        FileNotFoundError
            If the artifact holds no `.pt` weights file.
        """
        versioned_artifact_name = (
            f"{artifact.name.split(':')[0]}:{artifact.version}"
        )
        artifact_directory = Path(
            artifact.download(
                root=str(
                    self.download_directory
                    / versioned_artifact_name.replace(":", "-")
                )
            )
        )
        weight_files = sorted(artifact_directory.glob("*.pt"))
        if not weight_files:
            raise FileNotFoundError(
                f"Artifact '{artifact.name}' has no .pt weights file."
            )

        artifact_metadata = artifact.metadata
        model = SimpleAutoencoder(
            input_dim=int(artifact_metadata["window_size"]),
            hidden_dim=int(artifact_metadata["hidden_layer_size"]),
            latent_dim=int(artifact_metadata["latent_layer_size"]),
        )
        model.load_state_dict(
            torch.load(
                weight_files[0],
                map_location=self.device,
                weights_only=True,
            )
        )

        predictor = AutoencoderPredictor(
            model=model.to(self.device),
            metadata=AutoencoderModelMetadata(
                ticker=str(artifact_metadata["ticker"]),
                window_size=int(artifact_metadata["window_size"]),
                return_train_mean=float(
                    artifact_metadata["return_train_mean"]
                ),
                return_train_std=float(
                    artifact_metadata["return_train_std"]
                ),
                anomaly_threshold=float(
                    artifact_metadata["anomaly_threshold"]
                ),
                artifact_name=versioned_artifact_name,
            ),
            device=self.device,
        )
        self.models[predictor.metadata.ticker] = predictor
        return predictor


def _has_required_metadata(artifact: wandb.Artifact) -> bool:
    """Check whether an artifact can be used for inference.

    Parameters
    ----------
    artifact : wandb.Artifact
        Artifact to inspect.

    Returns
    -------
    bool
        `True` if the artifact metadata holds every key in
        `REQUIRED_METADATA_KEYS`.
    """
    return all(key in artifact.metadata for key in REQUIRED_METADATA_KEYS)
