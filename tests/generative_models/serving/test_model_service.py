"""Unit tests for AutoencoderModelService."""

import threading
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from generative_models.serving.model_service import (
    AutoencoderModelService,
    AvailableModel,
)

MODULE_PATH = "generative_models.serving.model_service"


@pytest.fixture(autouse=True)
def reset_service_instance() -> Iterator[None]:
    """Discard the singleton before and after each test.

    Returns
    -------
    Iterator[None]
        Generator used by pytest to run the teardown.
    """
    AutoencoderModelService.reset_instance()
    yield
    AutoencoderModelService.reset_instance()


def build_fake_predictor(ticker: str) -> MagicMock:
    """Create a fake predictor whose metadata holds a ticker.

    Parameters
    ----------
    ticker : str
        Ticker stored in the metadata.

    Returns
    -------
    MagicMock
        Fake predictor.
    """
    fake_predictor = MagicMock()
    fake_predictor.metadata.ticker = ticker
    return fake_predictor


def build_service() -> tuple[AutoencoderModelService, MagicMock]:
    """Create a service around a fake registry with real model storage.

    Returns
    -------
    tuple[AutoencoderModelService, MagicMock]
        The service and its fake registry.
    """
    fake_registry = MagicMock()
    fake_registry.models = {}
    fake_registry.get_model.side_effect = fake_registry.models.__getitem__
    return AutoencoderModelService(fake_registry), fake_registry


def fake_load_predictor(
    registry: MagicMock,
    ticker: str | None,
) -> MagicMock:
    """Store and return a fake predictor, as load_predictor would.

    Parameters
    ----------
    registry : MagicMock
        Fake registry whose `models` receives the predictor.
    ticker : str | None
        Requested ticker; `None` loads `"PETR4.SA"`.

    Returns
    -------
    MagicMock
        Fake predictor of the loaded ticker.
    """
    loaded_ticker = ticker or "PETR4.SA"
    fake_predictor = build_fake_predictor(loaded_ticker)
    registry.models[loaded_ticker] = fake_predictor
    return fake_predictor


def test_get_instance_returns_same_object() -> None:
    """CA-01: check that the singleton and its registry are reused."""
    with patch(f"{MODULE_PATH}.WandbAutoencoderRegistry") as mock_registry:
        first_service = AutoencoderModelService.get_instance()
        second_service = AutoencoderModelService.get_instance()

    assert first_service is second_service
    mock_registry.assert_called_once()


def test_get_instance_is_thread_safe() -> None:
    """CA-02: check that concurrent calls create a single instance."""
    received_services = []
    start_barrier = threading.Barrier(10)

    def get_service() -> None:
        start_barrier.wait()
        received_services.append(AutoencoderModelService.get_instance())

    with patch(f"{MODULE_PATH}.WandbAutoencoderRegistry") as mock_registry:
        threads = [threading.Thread(target=get_service) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert len(received_services) == 10
    assert all(
        service is received_services[0] for service in received_services
    )
    mock_registry.assert_called_once()


def test_get_instance_reads_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CA-03: check that the registry is configured from the environment."""
    monkeypatch.setenv("AUTOENCODER_WANDB_PROJECT", "env-project")
    monkeypatch.setenv("AUTOENCODER_WANDB_ENTITY", "env-entity")
    monkeypatch.setenv("AUTOENCODER_DEVICE", "cpu")

    with patch(f"{MODULE_PATH}.WandbAutoencoderRegistry") as mock_registry:
        AutoencoderModelService.get_instance()

    mock_registry.assert_called_once_with(
        project_name="env-project",
        entity="env-entity",
        device="cpu",
    )


def test_reset_instance_creates_new_instance() -> None:
    """CA-04: check that reset_instance discards the singleton."""
    with patch(f"{MODULE_PATH}.WandbAutoencoderRegistry"):
        first_service = AutoencoderModelService.get_instance()
        AutoencoderModelService.reset_instance()
        second_service = AutoencoderModelService.get_instance()

    assert first_service is not second_service


def test_get_predictor_without_ticker_loads_default_once() -> None:
    """CA-05: check that the default model is loaded a single time."""
    model_service, fake_registry = build_service()

    with patch(
        f"{MODULE_PATH}.load_predictor",
        side_effect=fake_load_predictor,
    ) as mock_load_predictor:
        first_predictor = model_service.get_predictor()
        second_predictor = model_service.get_predictor()

    assert first_predictor is second_predictor
    mock_load_predictor.assert_called_once_with(fake_registry, None)
    assert model_service.default_ticker == "PETR4.SA"


def test_get_predictor_reuses_model_in_memory() -> None:
    """CA-06: check that a loaded ticker is not downloaded again."""
    model_service, fake_registry = build_service()
    fake_predictor = build_fake_predictor("PETR4.SA")
    fake_registry.models["PETR4.SA"] = fake_predictor

    with patch(f"{MODULE_PATH}.load_predictor") as mock_load_predictor:
        predictor = model_service.get_predictor("petr4.sa")

    assert predictor is fake_predictor
    mock_load_predictor.assert_not_called()


def test_get_predictor_loads_missing_ticker_in_upper_case() -> None:
    """CA-07: check that a missing ticker is loaded in upper case."""
    model_service, fake_registry = build_service()

    with patch(
        f"{MODULE_PATH}.load_predictor",
        side_effect=fake_load_predictor,
    ) as mock_load_predictor:
        model_service.get_predictor("aapl")

    mock_load_predictor.assert_called_once_with(fake_registry, "AAPL")


def test_reload_without_ticker_sets_default() -> None:
    """CA-08: check that reload() defines the default ticker."""
    model_service, _ = build_service()

    with patch(
        f"{MODULE_PATH}.load_predictor",
        return_value=build_fake_predictor("VALE3.SA"),
    ):
        model_service.reload()

    assert model_service.default_ticker == "VALE3.SA"


def test_reload_with_ticker_keeps_default() -> None:
    """CA-09: check that reload(ticker) keeps the default ticker."""
    model_service, fake_registry = build_service()
    model_service.default_ticker = "PETR4.SA"

    with patch(
        f"{MODULE_PATH}.load_predictor",
        side_effect=fake_load_predictor,
    ) as mock_load_predictor:
        model_service.reload("aapl")

    mock_load_predictor.assert_called_once_with(fake_registry, "AAPL")
    assert model_service.default_ticker == "PETR4.SA"


def test_list_loaded_models_returns_metadata() -> None:
    """Check that the metadata of the models in memory is listed."""
    model_service, fake_registry = build_service()
    fake_registry.models["AAPL"] = build_fake_predictor("AAPL")
    fake_registry.models["PETR4.SA"] = build_fake_predictor("PETR4.SA")

    loaded_models = model_service.list_loaded_models()

    assert [metadata.ticker for metadata in loaded_models] == [
        "AAPL",
        "PETR4.SA",
    ]


def test_list_available_models_sorts_newest_first() -> None:
    """CA-10: check the order and versioned names of available models."""
    model_service, fake_registry = build_service()
    older_artifact = MagicMock()
    older_artifact.name = "simple-autoencoder-aapl:latest"
    older_artifact.version = "v1"
    older_artifact.created_at = "2026-08-01T00:00:00"
    older_artifact.metadata = {"ticker": "AAPL"}
    newer_artifact = MagicMock()
    newer_artifact.name = "simple-autoencoder-petr4.sa:latest"
    newer_artifact.version = "v4"
    newer_artifact.created_at = "2026-09-01T00:00:00"
    newer_artifact.metadata = {"ticker": "PETR4.SA"}
    fake_registry.list_available_artifacts.return_value = [
        older_artifact,
        newer_artifact,
    ]

    available_models = model_service.list_available_models()

    assert available_models == [
        AvailableModel(
            artifact_name="simple-autoencoder-petr4.sa:v4",
            ticker="PETR4.SA",
            created_at="2026-09-01T00:00:00",
        ),
        AvailableModel(
            artifact_name="simple-autoencoder-aapl:v1",
            ticker="AAPL",
            created_at="2026-08-01T00:00:00",
        ),
    ]
