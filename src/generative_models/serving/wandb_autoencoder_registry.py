"""Registry of autoencoder models stored as wandb artifacts."""

from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv

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
    """Registry that serves the autoencoders trained into wandb.

    The registry lists the model artifacts of a wandb project, keeps
    only the ones registered by the training pipeline, downloads their
    weights, rebuilds the `SimpleAutoencoder` with the architecture
    recorded in the artifact metadata and wraps it in an
    `AutoencoderPredictor` ready for inference.

    Models are kept in memory by ticker, so each one is downloaded
    only once per process.

    Parameters
    ----------
    project_name : str
        Name of the wandb project holding the model artifacts.
    entity : str | None
        Entity owning the project. When `None`, the default entity of
        the wandb client is used.
    download_directory : str
        Directory where artifacts are downloaded.
    env_file_path : str
        Path to the `.env` file holding the `WANDB_API_KEY` variable.
    device : str | None
        Device where the models are loaded. When `None`, `"cuda"` is
        used if available and `"cpu"` otherwise.

    Attributes
    ----------
    models : dict[str, AutoencoderPredictor]
        Predictors in memory, keyed by ticker.
    project_name : str
        Name of the wandb project.
    entity : str | None
        Entity owning the project, as received.
    download_directory : Path
        Directory where artifacts are downloaded.
    env_file_path : str
        Path to the `.env` file.
    device : str
        Device where the models are loaded.

    Example
    -------
    >>> registry = WandbAutoencoderRegistry()
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
        """Store the project settings without contacting wandb."""
        super().__init__()
        self.models: dict[str, AutoencoderPredictor] = {}
        self.project_name = project_name
        self.entity = entity
        self.download_directory = Path(download_directory)
        self.env_file_path = env_file_path

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._api: wandb.Api | None = None

    @property
    def api(self) -> wandb.Api:
        """Return the wandb client, building it on first use.

        Returns
        -------
        wandb.Api
            Client authenticated with the key read from the `.env`
            file. The same client is reused by later reads.
        """
        if self._api is None:
            load_dotenv(dotenv_path=self.env_file_path)
            self._api = wandb.Api()

        return self._api

    @property
    def project_path(self) -> str:
        """Return the `entity/project` path of the wandb project.

        Returns
        -------
        str
            Path built from `project_name` and either `entity` or the
            default entity of the client.
        """
        entity = self.entity
        if entity is None:
            entity = self.api.default_entity

        return f"{entity}/{self.project_name}"

    def list_available_artifacts(self) -> list[wandb.Artifact]:
        """List the artifacts of the project that can be served.

        Returns
        -------
        list[wandb.Artifact]
            The `latest` version of every model collection whose name
            starts with the training prefix and whose metadata holds
            every key inference requires. Nothing is downloaded.
        """
        project_path = self.project_path
        available_artifacts = []

        for collection in self.api.artifact_collections(
            project_name=project_path,
            type_name="model",
        ):
            if not collection.name.startswith(MODEL_NAME_PREFIX):
                continue

            artifact = self.api.artifact(
                f"{project_path}/{collection.name}:latest"
            )
            if _has_required_metadata(artifact):
                available_artifacts.append(artifact)

        return available_artifacts

    def load_latest_model(self) -> AutoencoderPredictor:
        """Load the most recently created servable model.

        Returns
        -------
        AutoencoderPredictor
            Predictor of the artifact with the newest `created_at`.

        Raises
        ------
        LookupError
            If the project holds no servable artifact.
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
        """Load the model of one ticker and keep it in memory.

        Parameters
        ----------
        ticker : str
            Ticker symbol whose latest artifact is loaded.

        Returns
        -------
        AutoencoderPredictor
            Predictor built from the latest artifact of `ticker`.

        Raises
        ------
        ValueError
            If the artifact metadata lacks a key inference requires.
        """
        artifact_slug = build_ticker_slug(ticker)
        artifact = self.api.artifact(
            f"{self.project_path}/{MODEL_NAME_PREFIX}{artifact_slug}:latest",
            type="model",
        )

        if not _has_required_metadata(artifact):
            raise ValueError(
                f"Artifact '{artifact.name}' is missing metadata "
                "required for inference. Expected keys: "
                f"{', '.join(REQUIRED_METADATA_KEYS)}."
            )

        return self._load_artifact(artifact)

    def _load_artifact(
        self,
        artifact: wandb.Artifact,
    ) -> AutoencoderPredictor:
        """Download an artifact and rebuild its predictor.

        Parameters
        ----------
        artifact : wandb.Artifact
            Artifact whose weights and metadata are read.

        Returns
        -------
        AutoencoderPredictor
            Predictor stored in `models` under the artifact ticker,
            replacing any previous predictor of the same ticker.

        Raises
        ------
        FileNotFoundError
            If the downloaded artifact holds no `.pt` file.
        """
        collection_name = artifact.name.split(":")[0]
        versioned_name = f"{collection_name}:{artifact.version}"
        download_root = self.download_directory / versioned_name.replace(
            ":", "-"
        )
        artifact.download(root=str(download_root))

        weights_files = sorted(Path(download_root).glob("*.pt"))
        if not weights_files:
            raise FileNotFoundError(
                f"Artifact '{versioned_name}' has no .pt weights file."
            )

        artifact_metadata = artifact.metadata
        model = SimpleAutoencoder(
            input_dim=int(artifact_metadata["window_size"]),
            hidden_dim=int(artifact_metadata["hidden_layer_size"]),
            latent_dim=int(artifact_metadata["latent_layer_size"]),
        )
        model.load_state_dict(
            torch.load(
                weights_files[0],
                map_location=self.device,
                weights_only=True,
            )
        )

        metadata = AutoencoderModelMetadata(
            ticker=str(artifact_metadata["ticker"]),
            window_size=int(artifact_metadata["window_size"]),
            return_train_mean=float(
                artifact_metadata["return_train_mean"]
            ),
            return_train_std=float(artifact_metadata["return_train_std"]),
            anomaly_threshold=float(
                artifact_metadata["anomaly_threshold"]
            ),
            artifact_name=versioned_name,
        )
        predictor = AutoencoderPredictor(
            model=model.to(self.device),
            metadata=metadata,
            device=self.device,
        )
        self.models[metadata.ticker] = predictor

        return predictor

    def load_all_models(self) -> list[str]:
        """Load every servable model of the project.

        Returns
        -------
        list[str]
            Ticker symbols of the loaded models, in the order the
            artifacts were listed.
        """
        return [
            self._load_artifact(artifact).metadata.ticker
            for artifact in self.list_available_artifacts()
        ]

    def get_model(self, ticker: str) -> AutoencoderPredictor:
        """Return a predictor already in memory.

        Parameters
        ----------
        ticker : str
            Ticker symbol of the predictor to retrieve.

        Returns
        -------
        AutoencoderPredictor
            Predictor held in memory for `ticker`. wandb is never
            contacted.

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
        """Return the tickers whose models are in memory.

        Returns
        -------
        list[str]
            Ticker symbols in the order their models were loaded.
        """
        return list(self.models)


def _has_required_metadata(artifact: wandb.Artifact) -> bool:
    """Check whether an artifact can be served for inference.

    Parameters
    ----------
    artifact : wandb.Artifact
        Artifact whose metadata is inspected.

    Returns
    -------
    bool
        `True` when the metadata holds every key of
        `REQUIRED_METADATA_KEYS`.
    """
    artifact_metadata = artifact.metadata or {}

    return all(
        metadata_key in artifact_metadata
        for metadata_key in REQUIRED_METADATA_KEYS
    )
