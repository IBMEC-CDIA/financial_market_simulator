"""Unit tests for AutoencoderModelService."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from generative_models.serving.model_service import AutoencoderModelService

MODULE_PATH = "generative_models.serving.model_service"


@pytest.fixture(autouse=True)
def reset_singleton():
    """Make sure every test starts and ends without a shared instance."""
    AutoencoderModelService.reset_instance()
    yield
    AutoencoderModelService.reset_instance()


def _make_predictor(ticker: str) -> SimpleNamespace:
    """Build a stand-in predictor that only carries metadata."""
    return SimpleNamespace(metadata=SimpleNamespace(ticker=ticker))


def _make_service() -> tuple[AutoencoderModelService, MagicMock]:
    """Build a service whose registry stores what load_predictor loads."""
    registry = MagicMock()
    registry.models = {}
    registry.get_model.side_effect = lambda ticker: registry.models[ticker]
    return AutoencoderModelService(registry), registry


@patch(f"{MODULE_PATH}.WandbAutoencoderRegistry")
def test_get_instance_returns_the_same_object(mock_registry_class) -> None:
    """Check that the service is created once and then reused."""
    first_instance = AutoencoderModelService.get_instance()
    second_instance = AutoencoderModelService.get_instance()

    assert first_instance is second_instance
    mock_registry_class.assert_called_once()


@patch(f"{MODULE_PATH}.WandbAutoencoderRegistry")
def test_get_instance_is_thread_safe(mock_registry_class) -> None:
    """Check that concurrent first calls still create one instance."""
    instances = []
    threads = [
        threading.Thread(
            target=lambda: instances.append(
                AutoencoderModelService.get_instance()
            )
        )
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len({id(instance) for instance in instances}) == 1
    mock_registry_class.assert_called_once()


@patch(f"{MODULE_PATH}.WandbAutoencoderRegistry")
def test_get_instance_reads_settings_from_environment(
    mock_registry_class, monkeypatch
) -> None:
    """Check that the registry is configured from the environment."""
    monkeypatch.setenv("AUTOENCODER_WANDB_PROJECT", "my-project")
    monkeypatch.setenv("AUTOENCODER_WANDB_ENTITY", "my-team")
    monkeypatch.setenv("AUTOENCODER_DEVICE", "cpu")

    AutoencoderModelService.get_instance()

    mock_registry_class.assert_called_once_with(
        project_name="my-project",
        entity="my-team",
        device="cpu",
    )


@patch(f"{MODULE_PATH}.load_predictor")
def test_get_predictor_loads_default_model_once(mock_load_predictor) -> None:
    """Check that the default model is downloaded only on first use."""
    model_service, registry = _make_service()
    predictor = _make_predictor("PETR4.SA")

    def load(registry_argument, _ticker):
        registry_argument.models[predictor.metadata.ticker] = predictor
        return predictor

    mock_load_predictor.side_effect = load

    assert model_service.get_predictor() is predictor
    assert model_service.get_predictor() is predictor
    mock_load_predictor.assert_called_once_with(registry, None)
    assert model_service.default_ticker == "PETR4.SA"


@patch(f"{MODULE_PATH}.load_predictor")
def test_get_predictor_reuses_loaded_ticker(mock_load_predictor) -> None:
    """Check that a ticker already in memory is not downloaded again."""
    model_service, registry = _make_service()
    predictor = _make_predictor("AAPL")
    registry.models["AAPL"] = predictor

    assert model_service.get_predictor("aapl") is predictor
    mock_load_predictor.assert_not_called()


@patch(f"{MODULE_PATH}.load_predictor")
def test_get_predictor_loads_missing_ticker(mock_load_predictor) -> None:
    """Check that an unknown ticker is loaded in upper case."""
    model_service, registry = _make_service()

    model_service.get_predictor("msft")

    mock_load_predictor.assert_called_once_with(registry, "MSFT")


@patch(f"{MODULE_PATH}.load_predictor")
def test_reload_without_ticker_updates_default(mock_load_predictor) -> None:
    """Check that reloading the latest model changes the default."""
    model_service, _ = _make_service()
    mock_load_predictor.return_value = _make_predictor("AAPL")

    model_service.reload()

    assert model_service.default_ticker == "AAPL"


@patch(f"{MODULE_PATH}.load_predictor")
def test_reload_with_ticker_keeps_default(mock_load_predictor) -> None:
    """Check that reloading a given ticker keeps the default model."""
    model_service, registry = _make_service()
    model_service.default_ticker = "PETR4.SA"
    mock_load_predictor.return_value = _make_predictor("AAPL")

    model_service.reload("aapl")

    mock_load_predictor.assert_called_once_with(registry, "AAPL")
    assert model_service.default_ticker == "PETR4.SA"


def test_list_available_models_sorts_newest_first() -> None:
    """Check the conversion and ordering of the wandb artifacts."""
    model_service, registry = _make_service()
    registry.list_available_artifacts.return_value = [
        SimpleNamespace(
            name="simple-autoencoder-aapl:latest",
            version="v1",
            metadata={"ticker": "AAPL"},
            created_at="2026-10-01T00:00:00Z",
        ),
        SimpleNamespace(
            name="simple-autoencoder-petr4.sa:latest",
            version="v0",
            metadata={"ticker": "PETR4.SA"},
            created_at="2026-10-05T00:00:00Z",
        ),
    ]

    available_models = model_service.list_available_models()

    assert [model.ticker for model in available_models] == [
        "PETR4.SA",
        "AAPL",
    ]
    assert available_models[1].artifact_name == "simple-autoencoder-aapl:v1"
