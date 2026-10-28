"""Process-wide service that keeps autoencoder predictors in memory."""

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
    """Summary of a servable artifact stored in wandb.

    Parameters
    ----------
    artifact_name : str
        Versioned artifact name (for example,
        `"simple-autoencoder-aapl:v3"`).
    ticker : str
        Ticker symbol of the model.
    created_at : str
        Creation timestamp of the artifact.
    """

    artifact_name: str
    ticker: str
    created_at: str


class AutoencoderModelService:
    """Thread-safe singleton that downloads each model only once.

    Parameters
    ----------
    registry : WandbAutoencoderRegistry
        Registry used to load and cache the predictors.

    Example
    -------
    >>> service = AutoencoderModelService.get_instance()
    >>> predictor = service.get_predictor("PETR4.SA")
    >>> service.get_predictor("petr4.sa") is predictor
    True
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self, registry: WandbAutoencoderRegistry) -> None:
        """Store the registry; no model is loaded yet."""
        self.registry = registry
        self.default_ticker = None
        self._lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "AutoencoderModelService":
        """Return the process-wide instance, creating it once.

        The registry is configured from the environment variables
        `AUTOENCODER_WANDB_PROJECT`, `AUTOENCODER_WANDB_ENTITY` and
        `AUTOENCODER_DEVICE`.

        Returns
        -------
        AutoencoderModelService
            Shared service instance.
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    registry = WandbAutoencoderRegistry(
                        project_name=os.getenv(
                            "AUTOENCODER_WANDB_PROJECT",
                            TrackingConfig.project_name,
                        ),
                        entity=os.getenv("AUTOENCODER_WANDB_ENTITY"),
                        device=os.getenv("AUTOENCODER_DEVICE"),
                    )
                    cls._instance = cls(registry)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Discard the shared instance."""
        with cls._instance_lock:
            cls._instance = None

    def get_predictor(
        self,
        ticker: str | None = None,
    ) -> AutoencoderPredictor:
        """Return a predictor, loading it only when not in memory.

        Parameters
        ----------
        ticker : str | None
            Ticker symbol (case-insensitive). `None` returns the
            default model, loading the latest one on first use.

        Returns
        -------
        AutoencoderPredictor
            Requested predictor.
        """
        with self._lock:
            if not ticker:
                if self.default_ticker is None:
                    return self.reload()
                return self.registry.get_model(self.default_ticker)

            normalized_ticker = ticker.upper()
            if normalized_ticker in self.registry.models:
                return self.registry.models[normalized_ticker]
            return load_predictor(self.registry, normalized_ticker)

    def reload(
        self,
        ticker: str | None = None,
    ) -> AutoencoderPredictor:
        """Download a model again, replacing the one in memory.

        Parameters
        ----------
        ticker : str | None
            Ticker symbol (case-insensitive). `None` loads the latest
            model of any ticker and makes it the default.

        Returns
        -------
        AutoencoderPredictor
            Freshly loaded predictor.
        """
        with self._lock:
            if not ticker:
                predictor = load_predictor(self.registry, None)
                self.default_ticker = predictor.metadata.ticker
                return predictor
            return load_predictor(self.registry, ticker.upper())

    def list_loaded_models(self) -> list[AutoencoderModelMetadata]:
        """List the metadata of the predictors in memory.

        Returns
        -------
        list[AutoencoderModelMetadata]
            Metadata in loading order.
        """
        with self._lock:
            return [
                predictor.metadata
                for predictor in self.registry.models.values()
            ]

    def list_available_models(self) -> list[AvailableModel]:
        """List the servable artifacts in wandb without downloading.

        Returns
        -------
        list[AvailableModel]
            Artifacts sorted from the newest to the oldest.
        """
        available_artifacts = sorted(
            self.registry.list_available_artifacts(),
            key=lambda artifact: str(artifact.created_at),
            reverse=True,
        )
        return [
            AvailableModel(
                artifact_name=(
                    f"{artifact.name.split(':')[0]}:{artifact.version}"
                ),
                ticker=str(artifact.metadata["ticker"]),
                created_at=str(artifact.created_at),
            )
            for artifact in available_artifacts
        ]
