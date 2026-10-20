"""Process-wide singleton that keeps autoencoder predictors in memory."""

import os
import threading
from dataclasses import dataclass

from generative_models.models.autoencoder.config import TrackingConfig
from generative_models.serving.autoencoder_predictor import (
    AutoencoderModelMetadata,
    AutoencoderPredictor,
)
from generative_models.serving.predict_autoencoder import load_predictor
from generative_models.serving.wandb_autoencoder_registry import (
    WandbAutoencoderRegistry,
)


@dataclass
class AvailableModel:
    """Summary of a servable model artifact stored in wandb.

    Parameters
    ----------
    artifact_name : str
        Artifact name and version (for example,
        `"simple-autoencoder-aapl:v3"`).
    ticker : str
        Ticker symbol the model was trained on.
    created_at : str
        Artifact creation timestamp, in ISO 8601 format (UTC).
    """

    artifact_name: str
    ticker: str
    created_at: str


class AutoencoderModelService:
    """Singleton that loads autoencoders from wandb once per process.

    Models are downloaded with `load_predictor`, the same function
    used by the `predict_autoencoder` script, and cached in a single
    `WandbAutoencoderRegistry`. Every API request then reuses the
    in-memory predictors instead of downloading the weights again.

    Use `get_instance` to obtain the shared instance; it is created on
    the first call, in a thread-safe way, with settings read from the
    environment:

    - `AUTOENCODER_WANDB_PROJECT`: wandb project with the artifacts
      (defaults to the project used by `AutoencoderTrainer`).
    - `AUTOENCODER_WANDB_ENTITY`: wandb entity (defaults to the
      entity of the API key).
    - `AUTOENCODER_DEVICE`: torch device (defaults to CUDA when
      available).

    Parameters
    ----------
    registry : WandbAutoencoderRegistry
        Registry that downloads and stores the predictors.

    Example
    -------
    >>> service = AutoencoderModelService.get_instance()
    >>> service is AutoencoderModelService.get_instance()
    True
    >>> predictor = service.get_predictor()
    >>> predictor.metadata.ticker
    'PETR4.SA'
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self, registry: WandbAutoencoderRegistry) -> None:
        """Store the registry and create the model access lock."""
        self.registry = registry
        self.default_ticker = None
        self._models_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "AutoencoderModelService":
        """Return the shared service, creating it on the first call.

        Returns
        -------
        AutoencoderModelService
            The single instance of the service in this process.
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(
                        WandbAutoencoderRegistry(
                            project_name=os.getenv(
                                "AUTOENCODER_WANDB_PROJECT",
                                TrackingConfig.project_name,
                            ),
                            entity=os.getenv("AUTOENCODER_WANDB_ENTITY"),
                            device=os.getenv("AUTOENCODER_DEVICE"),
                        )
                    )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Discard the shared instance, releasing the loaded models."""
        with cls._instance_lock:
            cls._instance = None

    def get_predictor(
        self,
        ticker: str | None = None,
    ) -> AutoencoderPredictor:
        """Get a predictor, loading it from wandb only if needed.

        Parameters
        ----------
        ticker : str | None
            Ticker whose model is requested (case-insensitive). When
            `None`, the default model is returned: the most recently
            trained one at the time it was first loaded.

        Returns
        -------
        AutoencoderPredictor
            In-memory predictor for the requested model.
        """
        with self._models_lock:
            if ticker is None:
                if self.default_ticker is None:
                    return self.reload()
                return self.registry.get_model(self.default_ticker)

            normalized_ticker = ticker.upper()
            if normalized_ticker in self.registry.models:
                return self.registry.get_model(normalized_ticker)
            return load_predictor(self.registry, normalized_ticker)

    def reload(
        self,
        ticker: str | None = None,
    ) -> AutoencoderPredictor:
        """Download the latest artifact again and replace it in memory.

        Parameters
        ----------
        ticker : str | None
            Ticker whose model is reloaded. When `None`, the most
            recently trained model of any ticker is loaded and becomes
            the new default model.

        Returns
        -------
        AutoencoderPredictor
            The freshly loaded predictor.
        """
        with self._models_lock:
            predictor = load_predictor(
                self.registry,
                ticker.upper() if ticker else None,
            )
            if ticker is None:
                self.default_ticker = predictor.metadata.ticker
            return predictor

    def list_loaded_models(self) -> list[AutoencoderModelMetadata]:
        """List the metadata of every model held in memory.

        Returns
        -------
        list[AutoencoderModelMetadata]
            Metadata of the loaded predictors, in loading order.
        """
        with self._models_lock:
            return [
                predictor.metadata
                for predictor in self.registry.models.values()
            ]

    def list_available_models(self) -> list[AvailableModel]:
        """List the latest servable artifact of each ticker in wandb.

        Returns
        -------
        list[AvailableModel]
            One entry per ticker, newest first.
        """
        available_models = [
            AvailableModel(
                artifact_name=(
                    f"{artifact.name.split(':')[0]}:{artifact.version}"
                ),
                ticker=str(artifact.metadata["ticker"]),
                created_at=str(artifact.created_at),
            )
            for artifact in self.registry.list_available_artifacts()
        ]
        return sorted(
            available_models,
            key=lambda available_model: available_model.created_at,
            reverse=True,
        )
