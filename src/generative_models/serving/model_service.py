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
    build_versioned_artifact_name,
)


@dataclass
class AvailableModel:
    """A model artifact that can be loaded from wandb.

    Parameters
    ----------
    artifact_name : str
        Versioned name of the artifact, for example
        `"simple-autoencoder-aapl:v3"`.
    ticker : str
        Ticker symbol the model was trained on.
    created_at : str
        Creation time of the artifact, as reported by wandb.
    """

    artifact_name: str
    ticker: str
    created_at: str


class AutoencoderModelService:
    """Share one model registry and its predictors across requests.

    Within a process, each model is downloaded from wandb only once and
    reused by every call. The service also keeps a default ticker,
    used when a request does not name one. The first instance is
    created by `get_instance`, and the settings come from the
    `AUTOENCODER_WANDB_PROJECT`, `AUTOENCODER_WANDB_ENTITY` and
    `AUTOENCODER_DEVICE` environment variables.

    Parameters
    ----------
    registry : WandbAutoencoderRegistry
        Registry that loads and keeps the models in memory.

    Attributes
    ----------
    registry : WandbAutoencoderRegistry
        Registry that keeps the loaded models.
    default_ticker : str | None
        Ticker of the default model, or `None` until it is loaded.

    Example
    -------
    >>> service = AutoencoderModelService.get_instance()
    >>> predictor = service.get_predictor("PETR4.SA")
    """

    _instance: "AutoencoderModelService | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, registry: WandbAutoencoderRegistry) -> None:
        """Store the registry and create the lock of the service."""
        self.registry = registry
        self.default_ticker: str | None = None
        self._lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "AutoencoderModelService":
        """Return the shared service, creating it on first use.

        Creation is thread-safe: concurrent callers receive the same
        instance.

        Returns
        -------
        AutoencoderModelService
            The process-wide service.
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(_build_registry_from_environment())

        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Discard the shared service so the next call creates a new one.

        This is mainly used by the tests.
        """
        with cls._instance_lock:
            cls._instance = None

    def get_predictor(
        self,
        ticker: str | None = None,
    ) -> AutoencoderPredictor:
        """Return the predictor of a ticker, loading it only if needed.

        Without a ticker, the default model is returned. It is loaded
        as the latest model when no default exists yet. With a ticker,
        the name is converted to upper case, and a model already in
        memory is reused.

        Parameters
        ----------
        ticker : str | None
            Ticker symbol. When empty or `None`, the default model is
            used.

        Returns
        -------
        AutoencoderPredictor
            Predictor ready for inference.
        """
        with self._lock:
            if not ticker:
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
        """Download a model again and replace the one in memory.

        Without a ticker, the latest model of any ticker is loaded and
        becomes the new default. With a ticker, only that model is
        replaced, and the default ticker does not change.

        Parameters
        ----------
        ticker : str | None
            Ticker symbol to reload. When empty or `None`, the latest
            model of any ticker is loaded.

        Returns
        -------
        AutoencoderPredictor
            The newly loaded predictor.
        """
        with self._lock:
            if not ticker:
                predictor = load_predictor(self.registry, None)
                self.default_ticker = predictor.metadata.ticker
                return predictor

            return load_predictor(self.registry, ticker.upper())

    def list_loaded_models(self) -> list[AutoencoderModelMetadata]:
        """List the metadata of the models currently in memory.

        Returns
        -------
        list[AutoencoderModelMetadata]
            Metadata of each loaded predictor, in loading order.
        """
        with self._lock:
            return [
                predictor.metadata
                for predictor in self.registry.models.values()
            ]

    def list_available_models(self) -> list[AvailableModel]:
        """List the model artifacts that can be loaded from wandb.

        Nothing is downloaded. The result is sorted from the most recent
        artifact to the oldest one.

        Returns
        -------
        list[AvailableModel]
            One entry per servable artifact, newest first.
        """
        artifacts = sorted(
            self.registry.list_available_artifacts(),
            key=lambda artifact: artifact.created_at,
            reverse=True,
        )

        return [
            AvailableModel(
                artifact_name=build_versioned_artifact_name(artifact),
                ticker=str(artifact.metadata["ticker"]),
                created_at=str(artifact.created_at),
            )
            for artifact in artifacts
        ]


def _build_registry_from_environment() -> WandbAutoencoderRegistry:
    """Create the registry from the `AUTOENCODER_*` environment variables.

    Returns
    -------
    WandbAutoencoderRegistry
        Registry configured with the project, entity and device found in
        the environment. Missing or empty variables use the defaults.
    """
    return WandbAutoencoderRegistry(
        project_name=(
            os.getenv("AUTOENCODER_WANDB_PROJECT")
            or TrackingConfig.project_name
        ),
        entity=os.getenv("AUTOENCODER_WANDB_ENTITY") or None,
        device=os.getenv("AUTOENCODER_DEVICE") or None,
    )
