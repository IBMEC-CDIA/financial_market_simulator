"""Unit tests for AutoencoderModelService."""

import threading
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from generative_models.serving.model_service import (
    AutoencoderModelService,
)

REGISTRY_PATH = (
    "generative_models.serving.model_service.WandbAutoencoderRegistry"
)
LOAD_PREDICTOR_PATH = "generative_models.serving.model_service.load_predictor"


@pytest.fixture(autouse=True)
def reset_service_instance():
    """Discard the shared service before and after every test."""
    AutoencoderModelService.reset_instance()
    yield
    AutoencoderModelService.reset_instance()


def _make_registry() -> MagicMock:
    """Build a fake registry that keeps predictors in a models dict."""
    registry = MagicMock()
    registry.models = {}
    registry.get_model.side_effect = lambda ticker: registry.models[ticker]

    return registry


def _make_artifact(
    collection_name: str,
    version: str,
    created_at: datetime,
    ticker: str,
) -> MagicMock:
    """Build a fake model artifact with the given creation time."""
    artifact = MagicMock()
    artifact.name = f"{collection_name}:{version}"
    artifact.version = version
    artifact.created_at = created_at
    artifact.metadata = {"ticker": ticker}

    return artifact


def test_get_instance_returns_the_same_object_and_builds_one_registry(
    monkeypatch,
) -> None:
    """Check that repeated calls reuse the single shared service."""
    monkeypatch.delenv("AUTOENCODER_WANDB_PROJECT", raising=False)
    with patch(REGISTRY_PATH) as registry_class:
        first_service = AutoencoderModelService.get_instance()
        second_service = AutoencoderModelService.get_instance()

    assert first_service is second_service
    registry_class.assert_called_once()


def test_get_instance_is_thread_safe() -> None:
    """Check that concurrent callers receive one instance and one registry."""
    thread_count = 10
    start_barrier = threading.Barrier(thread_count)
    received_services = []
    received_lock = threading.Lock()

    def request_service() -> None:
        """Wait for every thread, then ask for the shared service."""
        start_barrier.wait()
        service = AutoencoderModelService.get_instance()
        with received_lock:
            received_services.append(service)

    with patch(REGISTRY_PATH) as registry_class:
        threads = [
            threading.Thread(target=request_service)
            for _ in range(thread_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert len(received_services) == thread_count
    assert all(
        service is received_services[0] for service in received_services
    )
    registry_class.assert_called_once()


def test_get_instance_reads_settings_from_environment(monkeypatch) -> None:
    """Check that the environment variables configure the registry."""
    monkeypatch.setenv("AUTOENCODER_WANDB_PROJECT", "custom-project")
    monkeypatch.setenv("AUTOENCODER_WANDB_ENTITY", "custom-entity")
    monkeypatch.setenv("AUTOENCODER_DEVICE", "cpu")

    with patch(REGISTRY_PATH) as registry_class:
        AutoencoderModelService.get_instance()

    registry_class.assert_called_once_with(
        project_name="custom-project",
        entity="custom-entity",
        device="cpu",
    )


def test_reset_instance_makes_the_next_call_create_a_new_service() -> None:
    """Check that reset_instance discards the shared service."""
    with patch(REGISTRY_PATH):
        first_service = AutoencoderModelService.get_instance()
        AutoencoderModelService.reset_instance()
        second_service = AutoencoderModelService.get_instance()

    assert first_service is not second_service


def test_get_predictor_without_ticker_loads_the_default_once(
    make_predictor,
) -> None:
    """Check that the first call loads the latest model as the default."""
    registry = _make_registry()
    predictor = make_predictor()
    service = AutoencoderModelService(registry)

    with patch(LOAD_PREDICTOR_PATH, return_value=predictor) as load_mock:
        registry.models["PETR4.SA"] = predictor
        first_result = service.get_predictor()
        second_result = service.get_predictor()

    assert first_result is predictor
    assert second_result is predictor
    load_mock.assert_called_once_with(registry, None)
    assert service.default_ticker == "PETR4.SA"


def test_get_predictor_reuses_a_model_already_in_memory(
    make_predictor,
) -> None:
    """Check that a lowercase ticker finds the model loaded in memory."""
    registry = _make_registry()
    predictor = make_predictor(ticker="PETR4.SA")
    registry.models["PETR4.SA"] = predictor
    service = AutoencoderModelService(registry)

    with patch(LOAD_PREDICTOR_PATH) as load_mock:
        result = service.get_predictor("petr4.sa")

    assert result is predictor
    load_mock.assert_not_called()


def test_get_predictor_loads_an_unknown_ticker_in_uppercase(
    make_predictor,
) -> None:
    """Check that a missing ticker is loaded with its uppercase name."""
    registry = _make_registry()
    service = AutoencoderModelService(registry)

    with patch(
        LOAD_PREDICTOR_PATH,
        return_value=make_predictor(ticker="AAPL"),
    ) as load_mock:
        service.get_predictor("aapl")

    load_mock.assert_called_once_with(registry, "AAPL")


def test_reload_without_ticker_sets_the_default_ticker(make_predictor) -> None:
    """Check that reloading the latest model updates the default ticker."""
    registry = _make_registry()
    service = AutoencoderModelService(registry)

    with patch(
        LOAD_PREDICTOR_PATH,
        return_value=make_predictor(ticker="PETR4.SA"),
    ):
        service.reload()

    assert service.default_ticker == "PETR4.SA"


def test_reload_with_ticker_keeps_the_default_ticker(make_predictor) -> None:
    """Check that reloading one ticker does not change the default."""
    registry = _make_registry()
    service = AutoencoderModelService(registry)
    service.default_ticker = "PETR4.SA"

    with patch(
        LOAD_PREDICTOR_PATH,
        return_value=make_predictor(ticker="AAPL"),
    ) as load_mock:
        service.reload("aapl")

    load_mock.assert_called_once_with(registry, "AAPL")
    assert service.default_ticker == "PETR4.SA"


def test_list_loaded_models_returns_metadata_in_loading_order(
    make_predictor,
) -> None:
    """Check that the metadata of every loaded predictor is listed."""
    registry = _make_registry()
    registry.models["PETR4.SA"] = make_predictor(ticker="PETR4.SA")
    registry.models["AAPL"] = make_predictor(ticker="AAPL")
    service = AutoencoderModelService(registry)

    loaded_metadata = service.list_loaded_models()

    assert [metadata.ticker for metadata in loaded_metadata] == [
        "PETR4.SA",
        "AAPL",
    ]


def test_list_available_models_is_sorted_newest_first() -> None:
    """Check the order and the versioned names of the available models."""
    registry = _make_registry()
    registry.list_available_artifacts.return_value = [
        _make_artifact(
            "simple-autoencoder-petr4.sa",
            "v1",
            datetime(2026, 9, 1),
            "PETR4.SA",
        ),
        _make_artifact(
            "simple-autoencoder-aapl",
            "v4",
            datetime(2026, 9, 20),
            "AAPL",
        ),
    ]
    service = AutoencoderModelService(registry)

    available_models = service.list_available_models()

    assert [model.artifact_name for model in available_models] == [
        "simple-autoencoder-aapl:v4",
        "simple-autoencoder-petr4.sa:v1",
    ]
    assert available_models[0].ticker == "AAPL"
    assert available_models[0].created_at == str(datetime(2026, 9, 20))
    registry.load_model.assert_not_called()
    registry.load_latest_model.assert_not_called()
