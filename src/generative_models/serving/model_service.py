"""Process-wide service that keeps the served models in memory."""

import os
import threading
from dataclasses import dataclass
from typing import Optional

from generative_models.models.autoencoder.config import TrackingConfig
from generative_models.serving.autoencoder_predictor import (
    AutoencoderModelMetadata,
    AutoencoderPredictor,
)
from generative_models.serving.predict_autoencoder import load_predictor
from generative_models.serving.wandb_autoencoder_registry import (
    WandbAutoencoderRegistry,
)

PROJECT_ENVIRONMENT_VARIABLE = "AUTOENCODER_WANDB_PROJECT"
ENTITY_ENVIRONMENT_VARIABLE = "AUTOENCODER_WANDB_ENTITY"
DEVICE_ENVIRONMENT_VARIABLE = "AUTOENCODER_DEVICE"


@dataclass
class AvailableModel:
    """Summary of a model artifact that could be served.

    Parameters
    ----------
    artifact_name : str
        Versioned name of the artifact, such as
        `"simple-autoencoder-aapl:v3"`.
    ticker : str
        Ticker symbol stored in the artifact metadata.
    created_at : str
        Creation timestamp of the artifact.
    """

    artifact_name: str
    ticker: str
    created_at: str


class AutoencoderModelService:
    """Single source of served models inside one process.

    The service exists so that each model is downloaded from wandb
    only once, no matter how many requests ask for it. It also holds
    the notion of a default model, served whenever a request does not
    name a ticker.

    Every method that reads or fills the cache takes the same
    reentrant lock, so two simultaneous requests never download the
    same model twice.

    Parameters
    ----------
    registry : WandbAutoencoderRegistry
        Registry used to read the models and to cache them in memory.

    Attributes
    ----------
    registry : WandbAutoencoderRegistry
        Registry used to read the models.
    default_ticker : str | None
        Ticker of the default model, `None` until one is loaded.

    Example
    -------
    >>> service = AutoencoderModelService.get_instance()
    >>> predictor = service.get_predictor()
    >>> service.default_ticker
    'PETR4.SA'
    """

    _instance: Optional["AutoencoderModelService"] = None
    _instance_lock = threading.Lock()

    def __init__(self, registry: WandbAutoencoderRegistry) -> None:
        """Store the registry and start without a default model."""
        self.registry = registry
        self.default_ticker: str | None = None
        self._access_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "AutoencoderModelService":
        """Return the service of this process, building it once.

        Returns
        -------
        AutoencoderModelService
            The single instance of the service. The first caller
            builds it, under a lock with double checking, so that
            concurrent callers share one registry.
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(registry=cls._build_registry())

        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Discard the current instance, so the next call rebuilds it."""
        with cls._instance_lock:
            cls._instance = None

    @classmethod
    def _build_registry(cls) -> WandbAutoencoderRegistry:
        """Build the registry described by the environment variables.

        Returns
        -------
        WandbAutoencoderRegistry
            Registry whose project, entity and device come from
            `AUTOENCODER_WANDB_PROJECT`, `AUTOENCODER_WANDB_ENTITY`
            and `AUTOENCODER_DEVICE`, falling back to the defaults of
            the training configuration.
        """
        return WandbAutoencoderRegistry(
            project_name=os.getenv(
                PROJECT_ENVIRONMENT_VARIABLE,
                TrackingConfig.project_name,
            ),
            entity=os.getenv(ENTITY_ENVIRONMENT_VARIABLE),
            device=os.getenv(DEVICE_ENVIRONMENT_VARIABLE),
        )

    def get_predictor(
        self,
        ticker: str | None = None,
    ) -> AutoencoderPredictor:
        """Return a predictor, downloading it only when needed.

        Parameters
        ----------
        ticker : str | None
            Ticker symbol to serve, in any case. When `None`, the
            default model is served, and loaded first if no default
            has been chosen yet.

        Returns
        -------
        AutoencoderPredictor
            Predictor held in memory for the requested ticker.

        Example
        -------
        >>> predictor = service.get_predictor("petr4.sa")
        >>> predictor.metadata.ticker
        'PETR4.SA'
        """
        with self._access_lock:
            if not ticker:
                if self.default_ticker is None:
                    return self.reload()

                return self.registry.get_model(self.default_ticker)

            upper_case_ticker = ticker.upper()
            if upper_case_ticker in self.registry.models:
                return self.registry.models[upper_case_ticker]

            return load_predictor(self.registry, upper_case_ticker)

    def reload(
        self,
        ticker: str | None = None,
    ) -> AutoencoderPredictor:
        """Download a model again, replacing the one in memory.

        Parameters
        ----------
        ticker : str | None
            Ticker symbol to reload, in any case. When `None`, the
            most recent model of the project is loaded and becomes the
            new default; when given, the default model is left
            untouched.

        Returns
        -------
        AutoencoderPredictor
            The freshly downloaded predictor.
        """
        with self._access_lock:
            if not ticker:
                predictor = load_predictor(self.registry, None)
                self.default_ticker = predictor.metadata.ticker

                return predictor

            return load_predictor(self.registry, ticker.upper())

    def list_loaded_models(self) -> list[AutoencoderModelMetadata]:
        """List the metadata of every model in memory.

        Returns
        -------
        list[AutoencoderModelMetadata]
            Metadata of the cached predictors, in the order their
            models were loaded. wandb is never contacted.
        """
        with self._access_lock:
            return [
                predictor.metadata
                for predictor in self.registry.models.values()
            ]

    def list_available_models(self) -> list[AvailableModel]:
        """List the models the project could serve, newest first.

        Returns
        -------
        list[AvailableModel]
            One entry per servable artifact, sorted from the most
            recently created to the oldest. Nothing is downloaded.
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
            key=lambda model: model.created_at,
            reverse=True,
        )
