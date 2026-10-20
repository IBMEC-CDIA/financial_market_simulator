"""Unit tests for the autoencoder serving API."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from wandb.errors import CommError

from generative_models.models.autoencoder.autoencoder import SimpleAutoencoder
from generative_models.serving.api import app, get_model_service
from generative_models.serving.autoencoder_predictor import (
    AutoencoderModelMetadata,
    AutoencoderPredictor,
)
from generative_models.serving.model_service import AvailableModel

WINDOW_SIZE = 5


def _make_predictor(ticker: str = "PETR4.SA") -> AutoencoderPredictor:
    """Build a predictor around an untrained autoencoder."""
    torch.manual_seed(0)
    return AutoencoderPredictor(
        model=SimpleAutoencoder(input_dim=WINDOW_SIZE),
        metadata=AutoencoderModelMetadata(
            ticker=ticker,
            window_size=WINDOW_SIZE,
            return_train_mean=0.0,
            return_train_std=0.01,
            anomaly_threshold=1.0,
            artifact_name=f"simple-autoencoder-{ticker.lower()}:v0",
        ),
    )


@pytest.fixture(name="model_service")
def fixture_model_service() -> MagicMock:
    """Fake singleton service that serves an untrained predictor."""
    model_service = MagicMock()
    model_service.default_ticker = "PETR4.SA"
    model_service.get_predictor.side_effect = (
        lambda ticker=None: _make_predictor(ticker or "PETR4.SA")
    )
    model_service.list_loaded_models.return_value = [
        _make_predictor().metadata
    ]
    return model_service


@pytest.fixture(name="client")
def fixture_client(model_service: MagicMock):
    """Test client whose model service dependency is the fake service."""
    app.dependency_overrides[get_model_service] = lambda: model_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_startup_preloads_default_model(client, model_service) -> None:
    """Check that the default model is requested at startup."""
    assert client.get("/health").status_code == 200
    model_service.get_predictor.assert_any_call()


def test_health_reports_loaded_models(client) -> None:
    """Check the health payload."""
    response = client.get("/health")

    assert response.json() == {
        "status": "ok",
        "default_ticker": "PETR4.SA",
        "loaded_tickers": ["PETR4.SA"],
    }


def test_list_loaded_models(client) -> None:
    """Check that loaded model metadata is serialized."""
    response = client.get("/models/loaded")

    assert response.status_code == 200
    assert response.json()[0]["artifact_name"] == (
        "simple-autoencoder-petr4.sa:v0"
    )
    assert response.json()[0]["window_size"] == WINDOW_SIZE


def test_list_available_models(client, model_service) -> None:
    """Check that wandb artifacts are listed."""
    model_service.list_available_models.return_value = [
        AvailableModel(
            artifact_name="simple-autoencoder-aapl:v1",
            ticker="AAPL",
            created_at="2026-10-01T00:00:00Z",
        )
    ]

    response = client.get("/models/available")

    assert response.json() == [
        {
            "artifact_name": "simple-autoencoder-aapl:v1",
            "ticker": "AAPL",
            "created_at": "2026-10-01T00:00:00Z",
        }
    ]


def test_get_default_model(client) -> None:
    """Check that the default model metadata is returned."""
    response = client.get("/models/default")

    assert response.status_code == 200
    assert response.json()["ticker"] == "PETR4.SA"


def test_reload_model_with_ticker(client, model_service) -> None:
    """Check that the reload endpoint forwards the ticker."""
    model_service.reload.return_value = _make_predictor("AAPL")

    response = client.post("/models/reload", json={"ticker": "AAPL"})

    model_service.reload.assert_called_once_with("AAPL")
    assert response.json()["ticker"] == "AAPL"


def test_predict_from_log_returns(client, model_service) -> None:
    """Check the summary and windows of a log-return prediction."""
    log_returns = np.random.default_rng(0).normal(0, 0.01, 12).tolist()

    response = client.post(
        "/predictions/log-returns",
        json={"ticker": "AAPL", "log_returns": log_returns},
    )

    payload = response.json()
    assert response.status_code == 200
    model_service.get_predictor.assert_called_with("AAPL")
    assert payload["model"]["ticker"] == "AAPL"
    assert payload["number_of_windows"] == 12 - WINDOW_SIZE + 1
    assert payload["windows"][0]["window_end_index"] == WINDOW_SIZE - 1
    assert payload["windows"][0]["window_end_date"] is None
    assert payload["number_of_outliers"] == sum(
        window["is_outlier"] for window in payload["windows"]
    )


def test_predict_from_log_returns_too_short_returns_422(client) -> None:
    """Check that fewer values than the window size are rejected."""
    response = client.post(
        "/predictions/log-returns",
        json={"log_returns": [0.01, 0.02]},
    )

    assert response.status_code == 422
    assert "At least 5" in response.json()["detail"]


def test_predict_from_prices(client) -> None:
    """Check a prediction computed from closing prices."""
    prices = (100 * np.exp(np.cumsum(np.full(10, 0.001)))).tolist()

    response = client.post("/predictions/prices", json={"prices": prices})

    assert response.status_code == 200
    assert response.json()["number_of_windows"] == 9 - WINDOW_SIZE + 1


def test_predict_from_prices_rejects_non_positive_prices(client) -> None:
    """Check that request validation rejects non-positive prices."""
    response = client.post(
        "/predictions/prices",
        json={"prices": [10.0, 0.0, 11.0]},
    )

    assert response.status_code == 422


@patch("generative_models.serving.predict_autoencoder.fetch_price_series")
def test_predict_from_market_data(mock_fetch_price_series, client) -> None:
    """Check that market predictions carry dates and prices."""
    mock_fetch_price_series.return_value = pd.DataFrame(
        {"close_price": 100 * np.exp(np.cumsum(np.full(10, 0.001)))},
        index=pd.bdate_range("2026-09-01", periods=10),
    )

    response = client.get(
        "/predictions/market",
        params={
            "ticker": "PETR4.SA",
            "start_date": "2026-09-01",
            "end_date": "2026-09-30",
        },
    )

    payload = response.json()
    assert response.status_code == 200
    mock_fetch_price_series.assert_called_once_with(
        "PETR4.SA",
        "2026-09-01",
        "2026-09-30",
    )
    assert payload["number_of_windows"] == 9 - WINDOW_SIZE + 1
    assert payload["windows"][0]["window_end_index"] == WINDOW_SIZE - 1
    assert payload["windows"][-1]["window_end_date"] == "2026-09-14"
    assert payload["windows"][-1]["close_price"] == pytest.approx(
        100 * np.exp(0.01),
        rel=1e-5,
    )


def test_predict_from_market_data_rejects_inverted_period(client) -> None:
    """Check that start_date must be earlier than end_date."""
    response = client.get(
        "/predictions/market",
        params={"start_date": "2026-10-01", "end_date": "2026-01-01"},
    )

    assert response.status_code == 422


def test_unknown_model_returns_404(client, model_service) -> None:
    """Check that a missing wandb artifact becomes a 404."""
    model_service.get_predictor.side_effect = CommError(
        "artifact membership 'simple-autoencoder-xyz:latest' not found"
    )

    response = client.get("/models/default")

    assert response.status_code == 404


def test_wandb_outage_returns_503(client, model_service) -> None:
    """Check that other wandb errors become a 503."""
    model_service.list_available_models.side_effect = CommError("timeout")

    response = client.get("/models/available")

    assert response.status_code == 503


def test_missing_default_model_returns_404(client, model_service) -> None:
    """Check that an empty project becomes a 404."""
    model_service.get_predictor.side_effect = LookupError("No servable")

    response = client.get("/models/default")

    assert response.status_code == 404
    assert response.json()["detail"] == "No servable"


def test_openapi_documents_every_route(client) -> None:
    """Check that every route is documented in the OpenAPI schema."""
    openapi_paths = client.get("/openapi.json").json()["paths"]

    assert set(openapi_paths) == {
        "/health",
        "/models/loaded",
        "/models/available",
        "/models/default",
        "/models/reload",
        "/predictions/log-returns",
        "/predictions/prices",
        "/predictions/market",
    }
    for path_operations in openapi_paths.values():
        for operation in path_operations.values():
            assert operation["summary"]
            assert operation["description"]
