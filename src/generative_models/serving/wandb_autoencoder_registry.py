"""Model registry that loads autoencoders registered in Weights & Biases."""

from pathlib import Path

from dotenv import load_dotenv
import torch
import wandb

from generative_models.models.autoencoder.autoencoder import (
    SimpleAutoencoder,
)
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
    """Registry of autoencoders logged as wandb model artifacts.

    The artifacts are the ones the autoencoder trainer registers with
    `log_model`. Each model is downloaded, rebuilt with the architecture
    saved in its metadata and kept in memory as an
    `AutoencoderPredictor`.

    Parameters
    ----------
    project_name : str
        Name of the wandb project that holds the artifacts.
    entity : str | None
        wandb entity (user or team). When `None`, the default entity of
        the logged-in account is used.
    download_directory : str
        Local directory where the artifacts are downloaded.
    env_file_path : str
        Path to the `.env` file that contains `WANDB_API_KEY`. It is
        read the first time the wandb client is used.
    device : str | None
        Device where the models run. When `None`, `"cuda"` is used if
        available and `"cpu"` otherwise.

    Attributes
    ----------
    models : dict[str, AutoencoderPredictor]
        Predictors kept in memory, indexed by ticker.
    project_name : str
        Name of the wandb project.
    entity : str | None
        wandb entity, or `None` for the default entity.
    download_directory : Path
        Local directory where the artifacts are downloaded.
    env_file_path : str
        Path to the `.env` file.
    device : str
        Device where the models run.

    Example
    -------
    >>> registry = WandbAutoencoderRegistry(entity="my-team")
    >>> predictor = registry.load_model("PETR4.SA")
    >>> predictor.metadata.anomaly_threshold
    0.42
    """

    def __init__(
        self,
        project_name: str = TrackingConfig.project_name,
        entity: str | None = None,
        download_directory: str = "artifacts/wandb",
        env_file_path: str = ".env",
        device: str | None = None,
    ) -> None:
        """Store the settings. The wandb client is created lazily."""
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
        """Return the wandb client, creating it on first use.

        The `.env` file is loaded before the client is created, so
        `WANDB_API_KEY` can come from that file.

        Returns
        -------
        wandb.Api
            Client shared by every call of this registry.
        """
        if self._api is None:
            load_dotenv(dotenv_path=self.env_file_path)
            self._api = wandb.Api()

        return self._api

    def reset_api(self) -> None:
        """Discard the wandb client so the next call creates a new one.

        Used after the wandb connection is lost, so that a broken client
        is not reused by later requests.
        """
        self._api = None

    @property
    def project_path(self) -> str:
        """Return the project path in the form `<entity>/<project>`.

        Returns
        -------
        str
            Project path. The entity is the configured one, or the
            default entity of the wandb account when none is set.
        """
        entity = self.entity
        if entity is None:
            entity = self.api.default_entity

        return f"{entity}/{self.project_name}"

    def list_available_artifacts(self) -> list[wandb.Artifact]:
        """List the latest version of every servable autoencoder.

        Collections whose names do not start with the autoencoder
        prefix are skipped, and so are artifacts whose metadata lacks a
        key required for inference.

        Returns
        -------
        list[wandb.Artifact]
            Latest versions of the servable autoencoder artifacts.
        """
        servable_artifacts = []
        collections = self.api.artifact_collections(
            project_name=self.project_path,
            type_name="model",
        )
        for collection in collections:
            if not collection.name.startswith(MODEL_NAME_PREFIX):
                continue
            artifact = self.api.artifact(
                f"{self.project_path}/{collection.name}:latest",
                type="model",
            )
            if _find_missing_metadata_keys(artifact.metadata) == []:
                servable_artifacts.append(artifact)

        return servable_artifacts

    def load_latest_model(self) -> AutoencoderPredictor:
        """Load the most recently created servable autoencoder.

        Returns
        -------
        AutoencoderPredictor
            Predictor of the artifact with the greatest `created_at`.

        Raises
        ------
        LookupError
            If the project has no servable autoencoder artifact.
        """
        servable_artifacts = self.list_available_artifacts()
        if not servable_artifacts:
            raise LookupError(
                "No servable autoencoder artifact found in "
                f"'{self.project_path}'."
            )
        latest_artifact = max(
            servable_artifacts,
            key=lambda artifact: artifact.created_at,
        )

        return self._load_artifact(latest_artifact)

    def load_model(self, ticker: str) -> AutoencoderPredictor:
        """Load the latest model trained on a ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol, for example `"PETR4.SA"`.

        Returns
        -------
        AutoencoderPredictor
            Predictor of the latest artifact of the ticker.

        Raises
        ------
        ValueError
            If the artifact lacks metadata required for inference.
        wandb.errors.CommError
            If the artifact does not exist or wandb is unreachable.
        """
        artifact_name = (
            f"{self.project_path}/"
            f"{MODEL_NAME_PREFIX}{build_ticker_slug(ticker)}:latest"
        )
        artifact = self.api.artifact(artifact_name, type="model")
        missing_keys = _find_missing_metadata_keys(artifact.metadata)
        if missing_keys:
            raise ValueError(
                f"Artifact '{artifact_name}' is missing metadata required "
                f"for inference: {', '.join(missing_keys)}."
            )

        return self._load_artifact(artifact)

    def load_all_models(self) -> list[str]:
        """Load every servable autoencoder artifact of the project.

        Returns
        -------
        list[str]
            Tickers of the loaded models, in loading order.
        """
        return [
            self._load_artifact(artifact).metadata.ticker
            for artifact in self.list_available_artifacts()
        ]

    def get_model(self, ticker: str) -> AutoencoderPredictor:
        """Return the predictor of a ticker that is already in memory.

        Parameters
        ----------
        ticker : str
            Ticker symbol of a model loaded with `load_model`.

        Returns
        -------
        AutoencoderPredictor
            Predictor kept in memory for `ticker`.

        Raises
        ------
        KeyError
            If no model was loaded for `ticker`. The wandb API is not
            called.
        """
        if ticker not in self.models:
            raise KeyError(
                f"No model loaded for ticker '{ticker}'. "
                "Call load_model first."
            )

        return self.models[ticker]

    def list_loaded_tickers(self) -> list[str]:
        """List the tickers whose models are in memory.

        Returns
        -------
        list[str]
            Tickers in the order they were loaded.
        """
        return list(self.models.keys())

    def _load_artifact(self, artifact: wandb.Artifact) -> AutoencoderPredictor:
        """Download an artifact, rebuild its model and keep the predictor.

        Parameters
        ----------
        artifact : wandb.Artifact
            Model artifact registered by the autoencoder trainer.

        Returns
        -------
        AutoencoderPredictor
            Predictor kept in `models`, replacing any earlier predictor
            of the same ticker.

        Raises
        ------
        FileNotFoundError
            If the downloaded artifact contains no `.pt` file.
        """
        versioned_name = build_versioned_artifact_name(artifact)
        collection_name = artifact.name.split(":")[0]
        download_path = (
            self.download_directory / f"{collection_name}-{artifact.version}"
        )
        artifact.download(root=str(download_path))

        weight_paths = sorted(download_path.glob("*.pt"))
        if not weight_paths:
            raise FileNotFoundError(
                f"Artifact '{versioned_name}' has no .pt weights file."
            )

        metadata = artifact.metadata
        window_size = int(metadata["window_size"])
        model = SimpleAutoencoder(
            input_dim=window_size,
            hidden_dim=int(metadata["hidden_layer_size"]),
            latent_dim=int(metadata["latent_layer_size"]),
        )
        state_dict = torch.load(
            weight_paths[0],
            map_location=self.device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.to(self.device)

        predictor_metadata = AutoencoderModelMetadata(
            ticker=str(metadata["ticker"]),
            window_size=window_size,
            return_train_mean=float(metadata["return_train_mean"]),
            return_train_std=float(metadata["return_train_std"]),
            anomaly_threshold=float(metadata["anomaly_threshold"]),
            artifact_name=versioned_name,
        )
        predictor = AutoencoderPredictor(
            model=model,
            metadata=predictor_metadata,
            device=self.device,
        )
        self.models[predictor_metadata.ticker] = predictor

        return predictor


def build_versioned_artifact_name(artifact: wandb.Artifact) -> str:
    """Build the `<collection>:<version>` name of a model artifact.

    Parameters
    ----------
    artifact : wandb.Artifact
        Model artifact, whose name may carry an alias after `:`.

    Returns
    -------
    str
        Collection name followed by the artifact version, for example
        `"simple-autoencoder-aapl:v3"`.

    Example
    -------
    >>> build_versioned_artifact_name(artifact)
    'simple-autoencoder-aapl:v3'
    """
    collection_name = artifact.name.split(":")[0]

    return f"{collection_name}:{artifact.version}"


def _find_missing_metadata_keys(metadata: dict) -> list[str]:
    """List the required metadata keys that an artifact does not have.

    Parameters
    ----------
    metadata : dict
        Metadata saved with a model artifact.

    Returns
    -------
    list[str]
        Required keys absent from `metadata`, in the order of
        `REQUIRED_METADATA_KEYS`. An empty list means the artifact can
        be served.
    """
    return [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
