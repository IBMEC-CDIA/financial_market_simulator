"""Unit tests for AutoencoderModelService."""

import threading
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from generative_models.serving.model_service import AutoencoderModelService

MODULE_PATH = "generative_models.serving.model_service"


@pytest.fixture(autouse=True)
def reset_singleton() -> Iterator[None]:
    """Discard the shared instance before and after each test."""
    AutoencoderModelService.reset_instance()
    yield
    AutoencoderModelService.reset_instance()


def _make_predictor(ticker: str) -> MagicMock:
    """Build a fake predictor for a ticker."""
    predictor = MagicMock()
    predictor.metadata.ticker = ticker
    return predictor


def _make_service() -> AutoencoderModelService:
    """Build a service around a fake registry with no models."""
    registry = MagicMock()
    registry.models = {}
    return AutoencoderModelService(registry)


@patch(f"{MODULE_PATH}.WandbAutoencoderRegistry")
def test_get_instance_returns_same_object(mock_registry_class) -> None:
    """CA-01: the instance and its registry are created once."""
    first_service = AutoencoderModelService.get_instance()
    second_service = AutoencoderModelService.get_instance()

    assert first_service is second_service
    mock_registry_class.assert_called_once()


@patch(f"{MODULE_PATH}.WandbAutoencoderRegistry")
def test_get_instance_is_thread_safe(mock_registry_class) -> None:
    """CA-02: concurrent calls share a single instance."""
    services = []
    start_barrier = threading.Barrier(10)

    def get_service() -> None:
        """Wait for every thread, then get the instance."""
        start_barrier.wait()
        services.append(AutoencoderModelService.get_instance())

    threads = [threading.Thread(target=get_service) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len({id(service) for service in services}) == 1
    mock_registry_class.assert_called_once()


@patch(f"{MODULE_PATH}.WandbAutoencoderRegistry")
def test_get_instance_reads_environment(
    mock_registry_class,
    monkeypatch,
) -> None:
    """CA-03: the registry is configured from environment variables."""
    monkeypatch.setenv("AUTOENCODER_WANDB_PROJECT", "env-project")
    monkeypatch.setenv("AUTOENCODER_WANDB_ENTITY", "env-entity")
    monkeypatch.setenv("AUTOENCODER_DEVICE", "cpu")

    AutoencoderModelService.get_instance()

    mock_registry_class.assert_called_once_with(
        project_name="env-project",
        entity="env-entity",
        device="cpu",
    )


@patch(f"{MODULE_PATH}.WandbAutoencoderRegistry")
def test_reset_instance_creates_new_instance(mock_registry_class) -> None:
    """CA-04: a reset makes get_instance build a new instance."""
    first_service = AutoencoderModelService.get_instance()
    AutoencoderModelService.reset_instance()

    second_service = AutoencoderModelService.get_instance()

    assert first_service is not second_service
    assert mock_registry_class.call_count == 2


@patch(f"{MODULE_PATH}.load_predictor")
def test_default_predictor_is_loaded_once(mock_load_predictor) -> None:
    """CA-05: the default model is downloaded once and then reused."""
    service = _make_service()
    predictor = _make_predictor("PETR4.SA")
    mock_load_predictor.return_value = predictor
    service.registry.get_model.return_value = predictor

    first_predictor = service.get_predictor()
    second_predictor = service.get_predictor()

    assert first_predictor is second_predictor
    mock_load_predictor.assert_called_once_with(service.registry, None)
    assert service.default_ticker == "PETR4.SA"


@patch(f"{MODULE_PATH}.load_predictor")
def test_loaded_ticker_is_not_downloaded(mock_load_predictor) -> None:
    """CA-06: a ticker in memory is served without loading."""
    service = _make_service()
    predictor = _make_predictor("PETR4.SA")
    service.registry.models["PETR4.SA"] = predictor

    assert service.get_predictor("petr4.sa") is predictor
    mock_load_predictor.assert_not_called()


@patch(f"{MODULE_PATH}.load_predictor")
def test_missing_ticker_is_loaded_uppercase(mock_load_predictor) -> None:
    """CA-07: a ticker not in memory is loaded in uppercase."""
    service = _make_service()

    service.get_predictor("aapl")

    mock_load_predictor.assert_called_once_with(service.registry, "AAPL")


@patch(f"{MODULE_PATH}.load_predictor")
def test_reload_sets_default_ticker(mock_load_predictor) -> None:
    """CA-08: reload without ticker sets the default ticker."""
    service = _make_service()
    mock_load_predictor.return_value = _make_predictor("MSFT")

    service.reload()

    assert service.default_ticker == "MSFT"


@patch(f"{MODULE_PATH}.load_predictor")
def test_reload_ticker_keeps_default(mock_load_predictor) -> None:
    """CA-09: reloading a ticker keeps the previous default ticker."""
    service = _make_service()
    service.default_ticker = "PETR4.SA"
    mock_load_predictor.return_value = _make_predictor("AAPL")

    service.reload("aapl")

    mock_load_predictor.assert_called_once_with(service.registry, "AAPL")
    assert service.default_ticker == "PETR4.SA"


def test_list_available_models_newest_first() -> None:
    """CA-10: artifacts are sorted newest first with versioned names."""
    service = _make_service()
    older_artifact = MagicMock()
    older_artifact.name = "simple-autoencoder-aapl:latest"
    older_artifact.version = "v1"
    older_artifact.created_at = "2026-10-01T10:00:00"
    older_artifact.metadata = {"ticker": "AAPL"}
    newer_artifact = MagicMock()
    newer_artifact.name = "simple-autoencoder-msft:v4"
    newer_artifact.version = "v4"
    newer_artifact.created_at = "2026-10-05T10:00:00"
    newer_artifact.metadata = {"ticker": "MSFT"}
    service.registry.list_available_artifacts.return_value = [
        older_artifact,
        newer_artifact,
    ]

    available_models = service.list_available_models()

    assert [model.artifact_name for model in available_models] == [
        "simple-autoencoder-msft:v4",
        "simple-autoencoder-aapl:v1",
    ]
    assert [model.ticker for model in available_models] == ["MSFT", "AAPL"]
