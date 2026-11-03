"""Integration tests of the API with the models stored in a local directory."""

# pylint: disable=duplicate-code

import pytest
import torch
from fastapi.testclient import TestClient

from generative_models.models.base_autoencoder.autoencoder import (
    SimpleAutoencoder,
)
from generative_models.models.base_autoencoder.overfitting import (
    OverfittingReport,
    save_overfitting_report,
)
from generative_models.serving.api import app
from generative_models.serving.model_service import (
    AutoencoderModelService,
)

WINDOW_SIZE = 5


def _write_model(directory, ticker: str, name: str) -> None:
    """Write the weights and the overfitting report of a small model."""
    model = SimpleAutoencoder(
        input_dim=WINDOW_SIZE,
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
        window_size=WINDOW_SIZE,
        hidden_layer_size=4,
        latent_layer_size=2,
        return_train_mean=0.001,
        return_train_std=0.02,
        anomaly_threshold=1.25,
    )
    save_overfitting_report(report, directory / f"{name}-overfitting.json")


@pytest.fixture(name="api_client")
def _api_client(monkeypatch, tmp_path):
    """Start the API with two models read from a temporary directory."""
    _write_model(tmp_path, "AAPL", "simple-autoencoder-aapl")
    _write_model(tmp_path, "MSFT", "simple-autoencoder-msft")
    monkeypatch.setenv("AUTOENCODER_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("AUTOENCODER_DEVICE", "cpu")
    AutoencoderModelService.reset_instance()
    with TestClient(app) as client:
        yield client
    AutoencoderModelService.reset_instance()


def test_every_model_is_loaded_when_the_api_starts(api_client) -> None:
    """Check that the API keeps all the models of the directory in memory."""
    response = api_client.get("/models/loaded")

    assert response.status_code == 200
    assert sorted(item["ticker"] for item in response.json()) == [
        "AAPL",
        "MSFT",
    ]


def test_prediction_is_routed_by_ticker(api_client) -> None:
    """Check that the prediction uses the model of the requested ticker."""
    response = api_client.post(
        "/predictions/log-returns",
        json={"ticker": "MSFT", "log_returns": [0.0] * WINDOW_SIZE},
    )

    assert response.status_code == 200
    assert response.json()["model"]["ticker"] == "MSFT"
    assert response.json()["number_of_windows"] == 1


def test_unknown_ticker_returns_not_found(api_client) -> None:
    """Check that a ticker without a stored model is answered with 404."""
    response = api_client.post(
        "/predictions/log-returns",
        json={"ticker": "PETR4.SA", "log_returns": [0.0] * WINDOW_SIZE},
    )

    assert response.status_code == 404
