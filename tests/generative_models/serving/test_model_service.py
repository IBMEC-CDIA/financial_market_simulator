"""Unit tests for AutoencoderModelService."""

# pylint: disable=duplicate-code

import threading
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest
import torch

from generative_models.models.autoencoder.autoencoder import (
    SimpleAutoencoder,
)
from generative_models.serving.autoencoder_predictor import (
    AutoencoderModelMetadata,
    AutoencoderPredictor,
)
from generative_models.serving.model_service import (
    AutoencoderModelService,
)

SERVICE_MODULE = "generative_models.serving.model_service"


@pytest.fixture(autouse=True)
def reset_singleton() -> Iterator[None]:
    """Drop the singleton instance around every test."""
    AutoencoderModelService.reset_instance()
    yield
    AutoencoderModelService.reset_instance()


def build_predictor(ticker: str = "PETR4.SA") -> AutoencoderPredictor:
    """Build a predictor wrapping an untrained autoencoder.

    Parameters
    ----------
    ticker : str
        Ticker symbol stored in the predictor metadata.

    Returns
    -------
    AutoencoderPredictor
        Predictor on CPU, built from a fixed random seed.
    """
    torch.manual_seed(42)

    return AutoencoderPredictor(
        model=SimpleAutoencoder(input_dim=5),
        metadata=AutoencoderModelMetadata(
            ticker=ticker,
            window_size=5,
            return_train_mean=0.001,
            return_train_std=0.02,
            anomaly_threshold=0.5,
            artifact_name=f"simple-autoencoder-{ticker.lower()}:v1",
        ),
        device="cpu",
    )


def build_fake_registry() -> MagicMock:
    """Build a registry that caches predictors in a real dictionary.

    Returns
    -------
    MagicMock
        Stand-in registry whose `models` behaves like the real one.
    """
    registry = MagicMock()
    registry.models = {}
    registry.get_model.side_effect = (
        lambda ticker: registry.models[ticker]
    )

    return registry


def build_load_predictor(predictor: AutoencoderPredictor):
    """Build a replacement for `load_predictor` that fills the cache.

    Parameters
    ----------
    predictor : AutoencoderPredictor
        Predictor returned by every call.

    Returns
    -------
    Callable
        Function mirroring how the real registry caches a predictor.
    """

    def load_predictor(
        registry: MagicMock,
        _ticker: str | None = None,
    ) -> AutoencoderPredictor:
        registry.models[predictor.metadata.ticker] = predictor
        return predictor

    return load_predictor


def build_fake_artifact(
    collection_name: str,
    version: str,
    created_at: str,
    ticker: str,
) -> MagicMock:
    """Build a stand-in for a wandb model artifact.

    Parameters
    ----------
    collection_name : str
        Name of the artifact collection, without the version suffix.
    version : str
        Version label reported by the artifact.
    created_at : str
        Creation timestamp of the artifact.
    ticker : str
        Ticker symbol stored in the artifact metadata.

    Returns
    -------
    MagicMock
        Object exposing the attributes the service reads.
    """
    artifact = MagicMock()
    artifact.name = f"{collection_name}:latest"
    artifact.version = version
    artifact.created_at = created_at
    artifact.metadata = {"ticker": ticker}

    return artifact


def test_get_instance_returns_the_same_object() -> None:
    """CA-01: the singleton builds its registry exactly once."""
    with patch(f"{SERVICE_MODULE}.WandbAutoencoderRegistry") as registry:
        first_service = AutoencoderModelService.get_instance()
        second_service = AutoencoderModelService.get_instance()

    assert first_service is second_service
    registry.assert_called_once()


def test_get_instance_is_thread_safe() -> None:
    """CA-02: concurrent callers share one instance and one registry."""
    collected_services: list[Any] = []
    start_barrier = threading.Barrier(10)

    def call_get_instance() -> None:
        start_barrier.wait()
        collected_services.append(AutoencoderModelService.get_instance())

    with patch(f"{SERVICE_MODULE}.WandbAutoencoderRegistry") as registry:
        threads = [
            threading.Thread(target=call_get_instance) for _ in range(10)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert len(collected_services) == 10
    assert all(
        service is collected_services[0] for service in collected_services
    )
    registry.assert_called_once()


def test_registry_is_built_from_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CA-03: the three environment variables reach the registry."""
    monkeypatch.setenv("AUTOENCODER_WANDB_PROJECT", "env-project")
    monkeypatch.setenv("AUTOENCODER_WANDB_ENTITY", "env-entity")
    monkeypatch.setenv("AUTOENCODER_DEVICE", "cpu")

    with patch(f"{SERVICE_MODULE}.WandbAutoencoderRegistry") as registry:
        AutoencoderModelService.get_instance()

    keyword_arguments = registry.call_args.kwargs

    assert keyword_arguments["project_name"] == "env-project"
    assert keyword_arguments["entity"] == "env-entity"
    assert keyword_arguments["device"] == "cpu"


def test_reset_instance_forces_a_new_service() -> None:
    """CA-04: after a reset a brand new instance is created."""
    with patch(f"{SERVICE_MODULE}.WandbAutoencoderRegistry"):
        first_service = AutoencoderModelService.get_instance()
        AutoencoderModelService.reset_instance()
        second_service = AutoencoderModelService.get_instance()

    assert first_service is not second_service


def test_get_predictor_downloads_the_default_model_once() -> None:
    """CA-05: the default model is downloaded once and then reused."""
    predictor = build_predictor()
    service = AutoencoderModelService(registry=build_fake_registry())

    with patch(
        f"{SERVICE_MODULE}.load_predictor",
        side_effect=build_load_predictor(predictor),
    ) as load_mock:
        first_predictor = service.get_predictor()
        second_predictor = service.get_predictor()

    assert first_predictor is predictor
    assert second_predictor is predictor
    load_mock.assert_called_once_with(service.registry, None)
    assert service.default_ticker == "PETR4.SA"


def test_get_predictor_reuses_a_model_already_in_memory() -> None:
    """CA-06: a cached ticker is served without touching wandb."""
    predictor = build_predictor()
    registry = build_fake_registry()
    registry.models["PETR4.SA"] = predictor
    service = AutoencoderModelService(registry=registry)

    with patch(f"{SERVICE_MODULE}.load_predictor") as load_mock:
        served_predictor = service.get_predictor("petr4.sa")

    assert served_predictor is predictor
    load_mock.assert_not_called()


def test_get_predictor_loads_an_unknown_ticker_in_upper_case() -> None:
    """CA-07: an unknown ticker is loaded, upper-cased."""
    predictor = build_predictor(ticker="AAPL")
    service = AutoencoderModelService(registry=build_fake_registry())

    with patch(
        f"{SERVICE_MODULE}.load_predictor",
        side_effect=build_load_predictor(predictor),
    ) as load_mock:
        service.get_predictor("aapl")

    load_mock.assert_called_once_with(service.registry, "AAPL")


def test_reload_without_ticker_sets_the_default() -> None:
    """CA-08: reloading without a ticker redefines the default one."""
    predictor = build_predictor()
    service = AutoencoderModelService(registry=build_fake_registry())

    with patch(
        f"{SERVICE_MODULE}.load_predictor",
        side_effect=build_load_predictor(predictor),
    ):
        service.reload()

    assert service.default_ticker == "PETR4.SA"


def test_reload_with_ticker_keeps_the_default() -> None:
    """CA-09: reloading one ticker never changes the default model."""
    default_predictor = build_predictor()
    service = AutoencoderModelService(registry=build_fake_registry())

    with patch(
        f"{SERVICE_MODULE}.load_predictor",
        side_effect=build_load_predictor(default_predictor),
    ):
        service.reload()

    other_predictor = build_predictor(ticker="AAPL")
    with patch(
        f"{SERVICE_MODULE}.load_predictor",
        side_effect=build_load_predictor(other_predictor),
    ) as load_mock:
        service.reload("aapl")

    load_mock.assert_called_once_with(service.registry, "AAPL")
    assert service.default_ticker == "PETR4.SA"


def test_list_available_models_is_sorted_by_recency() -> None:
    """CA-10: available models come newest first with versioned names."""
    registry = build_fake_registry()
    registry.list_available_artifacts.return_value = [
        build_fake_artifact(
            "simple-autoencoder-petr4.sa",
            "v1",
            "2026-01-01T00:00:00",
            "PETR4.SA",
        ),
        build_fake_artifact(
            "simple-autoencoder-aapl",
            "v3",
            "2026-05-01T00:00:00",
            "AAPL",
        ),
    ]
    service = AutoencoderModelService(registry=registry)

    available_models = service.list_available_models()

    assert [model.ticker for model in available_models] == [
        "AAPL",
        "PETR4.SA",
    ]
    assert available_models[0].artifact_name == (
        "simple-autoencoder-aapl:v3"
    )
    assert available_models[0].created_at == "2026-05-01T00:00:00"
