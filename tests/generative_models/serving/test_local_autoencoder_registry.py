"""Unit tests for the registry that serves models stored on disk."""

import os
from unittest.mock import MagicMock

import pytest
import torch

from generative_models.models.base_autoencoder.autoencoder import (
    SimpleAutoencoder,
)
from generative_models.models.base_autoencoder.overfitting import (
    OverfittingReport,
    save_overfitting_report,
)
from generative_models.serving.local_autoencoder_registry import (
    LocalAutoencoderRegistry,
)
from generative_models.serving.model_service import (
    AutoencoderModelService,
    _build_registry_from_environment,
)

WINDOW_SIZE = 5


def _write_model(
    directory,
    ticker: str,
    name: str,
    window_size: int = WINDOW_SIZE,
) -> None:
    """Write the weights and the overfitting report of a small model."""
    model = SimpleAutoencoder(
        input_dim=window_size,
        hidden_dim=4,
        latent_dim=2,
    )
    torch.save(model.state_dict(), directory / f"{name}.pt")
    report = OverfittingReport(
        ticker=ticker,
        model_file_path=f"artifacts/{name}.pt",
        final_train_loss=0.5,
        train_evaluation_loss=0.4,
        validation_loss=0.6,
        overfitting_ratio=1.5,
        test_loss=0.3,
        baseline_test_loss=0.9,
        window_size=window_size,
        hidden_layer_size=4,
        latent_layer_size=2,
        return_train_mean=0.001,
        return_train_std=0.02,
        anomaly_threshold=1.25,
    )
    save_overfitting_report(report, directory / f"{name}-overfitting.json")


def test_load_model_rebuilds_the_predictor_of_a_ticker(tmp_path) -> None:
    """Check that a stored model is loaded by its ticker."""
    _write_model(tmp_path, "AAPL", "simple-autoencoder-aapl")
    registry = LocalAutoencoderRegistry(str(tmp_path), device="cpu")

    predictor = registry.load_model("aapl")

    assert predictor.metadata.ticker == "AAPL"
    assert predictor.metadata.window_size == WINDOW_SIZE
    assert predictor.metadata.anomaly_threshold == pytest.approx(1.25)
    assert registry.get_model("AAPL") is predictor


def test_load_all_models_keeps_every_ticker_in_memory(tmp_path) -> None:
    """Check that every complete model is loaded at once."""
    _write_model(tmp_path, "AAPL", "simple-autoencoder-aapl")
    _write_model(tmp_path, "MSFT", "simple-autoencoder-msft")
    registry = LocalAutoencoderRegistry(str(tmp_path), device="cpu")

    tickers = registry.load_all_models()

    assert sorted(tickers) == ["AAPL", "MSFT"]
    assert sorted(registry.list_loaded_tickers()) == ["AAPL", "MSFT"]


def test_load_latest_model_uses_the_newest_weights(tmp_path) -> None:
    """Check that the model with the most recent weights is loaded."""
    _write_model(tmp_path, "AAPL", "simple-autoencoder-aapl")
    _write_model(tmp_path, "MSFT", "simple-autoencoder-msft")
    os.utime(tmp_path / "simple-autoencoder-aapl.pt", (1_000, 1_000))
    os.utime(tmp_path / "simple-autoencoder-msft.pt", (2_000, 2_000))
    registry = LocalAutoencoderRegistry(str(tmp_path), device="cpu")

    predictor = registry.load_latest_model()

    assert predictor.metadata.ticker == "MSFT"


def test_incomplete_models_are_not_listed(tmp_path) -> None:
    """Check that a report without weights, or without serving values, is
    skipped."""
    _write_model(tmp_path, "AAPL", "simple-autoencoder-aapl")
    (tmp_path / "simple-autoencoder-msft.pt").write_bytes(b"weights")
    _write_model(tmp_path, "PETR4.SA", "simple-autoencoder-petr4")
    (tmp_path / "simple-autoencoder-petr4.pt").unlink()
    (tmp_path / "simple-autoencoder-petr4-overfitting.json").write_text(
        '{"ticker": "PETR4.SA"}', encoding="utf-8"
    )
    registry = LocalAutoencoderRegistry(str(tmp_path), device="cpu")

    names = [artifact.name for artifact in registry.list_available_artifacts()]

    assert names == ["simple-autoencoder-aapl"]


def test_unknown_ticker_raises_lookup_error(tmp_path) -> None:
    """Check that a ticker without a stored model raises LookupError."""
    _write_model(tmp_path, "AAPL", "simple-autoencoder-aapl")
    registry = LocalAutoencoderRegistry(str(tmp_path), device="cpu")

    with pytest.raises(LookupError, match="PETR4.SA"):
        registry.load_model("PETR4.SA")


def test_missing_directory_has_no_models(tmp_path) -> None:
    """Check that a directory that does not exist lists no models."""
    registry = LocalAutoencoderRegistry(str(tmp_path / "absent"), device="cpu")

    assert not registry.list_available_artifacts()
    with pytest.raises(LookupError):
        registry.load_latest_model()


def test_get_model_requires_a_previous_load(tmp_path) -> None:
    """Check that get_model does not read the disk for a missing ticker."""
    registry = LocalAutoencoderRegistry(str(tmp_path), device="cpu")

    with pytest.raises(KeyError, match="load_model"):
        registry.get_model("AAPL")


def test_environment_selects_local_registry(monkeypatch, tmp_path) -> None:
    """Check that AUTOENCODER_MODEL_DIR switches the service to local files."""
    monkeypatch.setenv("AUTOENCODER_MODEL_DIR", str(tmp_path))

    registry = _build_registry_from_environment()

    assert isinstance(registry, LocalAutoencoderRegistry)
    assert registry.model_directory == tmp_path


def test_preload_all_asks_the_registry_for_every_model() -> None:
    """Check that the service loads every model through the registry."""
    registry = MagicMock()
    registry.load_all_models.return_value = ["AAPL", "MSFT"]
    service = AutoencoderModelService(registry=registry)

    assert service.preload_all() == ["AAPL", "MSFT"]
    registry.load_all_models.assert_called_once_with()
