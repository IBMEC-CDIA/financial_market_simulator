"""Process-wide service that keeps the served autoencoders in memory."""

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
    """Summary of a servable model artifact registered in wandb.

    Parameters
    ----------
    artifact_name : str
        Versioned artifact name, for example
        `"simple-autoencoder-aapl:v3"`.
    ticker : str
        Ticker symbol the model was trained on.
    created_at : str
        Creation timestamp of the artifact.
    """

    artifact_name: str
    ticker: str
    created_at: str


class AutoencoderModelService:
    """Singleton that loads each model once and shares it per process.

    The service wraps a `WandbAutoencoderRegistry`, whose `models`
    dictionary is the in-memory cache, and tracks the default model
    used when a request does not name a ticker. All loading methods
    share a reentrant lock, so concurrent requests never download the
    same model twice.

    Parameters
    ----------
    registry : WandbAutoencoderRegistry
        Registry used to load and keep the models.

    Example
    -------
    >>> model_service = AutoencoderModelService.get_instance()
    >>> predictor = model_service.get_predictor("PETR4.SA")
    >>> model_service.get_predictor("petr4.sa") is predictor
    True
    """

    _instance: "AutoencoderModelService | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, registry: WandbAutoencoderRegistry) -> None:
        """Store the registry; no model is loaded yet."""
        self.registry = registry
        self.default_ticker: str | None = None
        self._loading_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "AutoencoderModelService":
        """Return the process-wide service, creating it once.

        The registry is configured from the environment variables
        `AUTOENCODER_WANDB_PROJECT`, `AUTOENCODER_WANDB_ENTITY` and
        `AUTOENCODER_DEVICE`.

        Returns
        -------
        AutoencoderModelService
            The same instance on every call until `reset_instance`.
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
        """Discard the process-wide service."""
        with cls._instance_lock:
            cls._instance = None

    def get_predictor(
        self,
        ticker: str | None = None,
    ) -> AutoencoderPredictor:
        """Return a predictor, loading it only if not yet in memory.

        Parameters
        ----------
        ticker : str | None
            Ticker symbol, case insensitive. `None` returns the default
            model, loading the most recent one on the first call.

        Returns
        -------
        AutoencoderPredictor
            Predictor of the requested ticker or the default model.
        """
        with self._loading_lock:
            if not ticker:
                if self.default_ticker is None:
                    return self.reload()
                return self.registry.get_model(self.default_ticker)

            upper_ticker = ticker.upper()
            if upper_ticker in self.registry.models:
                return self.registry.models[upper_ticker]
            return load_predictor(self.registry, upper_ticker)

    def reload(
        self,
        ticker: str | None = None,
    ) -> AutoencoderPredictor:
        """Download a model again, replacing the one in memory.

        Parameters
        ----------
        ticker : str | None
            Ticker symbol, case insensitive. `None` loads the most
            recent model of any ticker and makes it the default.

        Returns
        -------
        AutoencoderPredictor
            Freshly loaded predictor.
        """
        with self._loading_lock:
            if not ticker:
                predictor = load_predictor(self.registry, None)
                self.default_ticker = predictor.metadata.ticker
                return predictor
            return load_predictor(self.registry, ticker.upper())

    def list_loaded_models(self) -> list[AutoencoderModelMetadata]:
        """List the metadata of the models in memory.

        Returns
        -------
        list[AutoencoderModelMetadata]
            Metadata of each loaded predictor, in loading order.
        """
        with self._loading_lock:
            return [
                predictor.metadata
                for predictor in self.registry.models.values()
            ]

    def list_available_models(self) -> list[AvailableModel]:
        """List the servable artifacts in wandb without downloading them.

        Returns
        -------
        list[AvailableModel]
            One entry per artifact, from the newest to the oldest.
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
