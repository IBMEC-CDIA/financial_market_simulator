"""Unit tests for the autoencoder REST API."""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import wandb
from fastapi.testclient import TestClient

from generative_models.models.autoencoder.autoencoder import SimpleAutoencoder
from generative_models.serving.api import app, get_model_service
from generative_models.serving.autoencoder_predictor import (
    AutoencoderModelMetadata,
    AutoencoderPredictor,
)
from generative_models.serving.model_service import AvailableModel

FETCH_PATH = "generative_models.serving.predict_autoencoder.fetch_price_series"


def _make_predictor() -> AutoencoderPredictor:
    """Build a predictor around an untrained autoencoder."""
    torch.manual_seed(0)
    return AutoencoderPredictor(
        model=SimpleAutoencoder(input_dim=5),
        metadata=AutoencoderModelMetadata(
            ticker="PETR4.SA",
            window_size=5,
            return_train_mean=0.0,
            return_train_std=0.02,
            anomaly_threshold=1.0,
            artifact_name="simple-autoencoder-petr4.sa:v0",
        ),
    )


@pytest.fixture(name="model_service")
def fixture_model_service() -> Iterator[MagicMock]:
    """Override the model service with a mock serving one predictor."""
    predictor = _make_predictor()
    model_service = MagicMock()
    model_service.default_ticker = "PETR4.SA"
    model_service.get_predictor.return_value = predictor
    model_service.reload.return_value = predictor
    model_service.list_loaded_models.return_value = [predictor.metadata]
    app.dependency_overrides[get_model_service] = lambda: model_service
    yield model_service
    app.dependency_overrides.clear()


@pytest.fixture(name="client")
def fixture_client(model_service: MagicMock) -> Iterator[TestClient]:
    """Run the application with the mocked service."""
    with TestClient(app) as test_client:
        model_service.get_predictor.reset_mock()
        yield test_client


def test_startup_preloads_default_model(model_service) -> None:
    """CA-01: startup calls get_predictor without arguments."""
    with TestClient(app):
        model_service.get_predictor.assert_called_once_with()


def test_startup_skips_preload(model_service, monkeypatch) -> None:
    """CA-02: AUTOENCODER_PRELOAD_MODEL=false skips preloading."""
    monkeypatch.setenv("AUTOENCODER_PRELOAD_MODEL", "False")

    with TestClient(app):
        model_service.get_predictor.assert_not_called()


def test_startup_survives_preload_failure(model_service) -> None:
    """CA-03: a preload failure does not stop the API."""
    model_service.get_predictor.side_effect = RuntimeError("offline")

    with TestClient(app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200


def test_health(client) -> None:
    """CA-04: /health reports the default and loaded tickers."""
    response = client.get("/health")

    assert response.json() == {
        "status": "ok",
        "default_ticker": "PETR4.SA",
        "loaded_tickers": ["PETR4.SA"],
    }


def test_loaded_models(client) -> None:
    """CA-05: /models/loaded returns the in-memory metadata."""
    response = client.get("/models/loaded")

    assert response.status_code == 200
    assert response.json() == [
        {
            "ticker": "PETR4.SA",
            "artifact_name": "simple-autoencoder-petr4.sa:v0",
            "window_size": 5,
            "return_train_mean": 0.0,
            "return_train_std": 0.02,
            "anomaly_threshold": 1.0,
        }
    ]


def test_available_models(client, model_service) -> None:
    """CA-06: /models/available keeps the service order."""
    model_service.list_available_models.return_value = [
        AvailableModel("simple-autoencoder-msft:v4", "MSFT", "2026-10-05"),
        AvailableModel("simple-autoencoder-aapl:v1", "AAPL", "2026-10-01"),
    ]

    response = client.get("/models/available")

    assert [model["ticker"] for model in response.json()] == [
        "MSFT",
        "AAPL",
    ]


def test_default_model(client) -> None:
    """CA-07: /models/default returns the default model metadata."""
    response = client.get("/models/default")

    assert response.status_code == 200
    assert response.json()["ticker"] == "PETR4.SA"


def test_reload(client, model_service) -> None:
    """CA-08: /models/reload forwards the optional ticker."""
    client.post("/models/reload", json={"ticker": "AAPL"})
    client.post("/models/reload", json={})

    assert [call.args for call in model_service.reload.call_args_list] == [
        ("AAPL",),
        (None,),
    ]


def test_predict_log_returns(client, model_service) -> None:
    """CA-09: 12 log-returns give 8 windows without dates."""
    log_returns = np.random.default_rng(0).normal(0, 0.02, 12).tolist()

    response = client.post(
        "/predictions/log-returns",
        json={"ticker": "PETR4.SA", "log_returns": log_returns},
    )

    windows = response.json()["windows"]
    assert response.status_code == 200
    assert len(windows) == 8
    assert [window["window_end_index"] for window in windows] == list(
        range(4, 12)
    )
    assert all(window["window_end_date"] is None for window in windows)
    model_service.get_predictor.assert_called_once_with("PETR4.SA")


def test_predict_log_returns_too_short(client) -> None:
    """CA-10: fewer log-returns than window_size respond 422."""
    response = client.post(
        "/predictions/log-returns", json={"log_returns": [0.01, 0.02]}
    )

    assert response.status_code == 422


def test_predict_prices(client) -> None:
    """CA-11: 12 prices give 7 windows."""
    prices = (100 + np.arange(12, dtype=float)).tolist()

    response = client.post("/predictions/prices", json={"prices": prices})

    assert response.status_code == 200
    assert response.json()["number_of_windows"] == 7


def test_predict_prices_rejects_zero(client) -> None:
    """CA-12: a zero price responds 422."""
    response = client.post(
        "/predictions/prices", json={"prices": [10.0, 0.0, 11.0]}
    )

    assert response.status_code == 422


@patch(FETCH_PATH)
def test_predict_market(mock_fetch, client) -> None:
    """CA-13: market data is fetched and aligned with dates and prices."""
    price_data = pd.DataFrame(
        {"close_price": 100 + np.arange(10, dtype=float)},
        index=pd.bdate_range("2026-09-01", periods=10),
    )
    mock_fetch.return_value = price_data

    response = client.get(
        "/predictions/market",
        params={"start_date": "2026-09-01", "end_date": "2026-09-30"},
    )

    windows = response.json()["windows"]
    mock_fetch.assert_called_once_with(
        "PETR4.SA", "2026-09-01", "2026-09-30"
    )
    assert len(windows) == 5
    assert windows[0]["window_end_index"] == 4
    assert windows[-1]["window_end_date"] == "2026-09-14"
    assert windows[-1]["close_price"] == price_data["close_price"].iloc[-1]


def test_predict_market_invalid_period(client) -> None:
    """CA-14: start_date after end_date responds 422."""
    response = client.get(
        "/predictions/market",
        params={"start_date": "2026-09-30", "end_date": "2026-09-01"},
    )

    assert response.status_code == 422


def test_wandb_errors_are_translated(client, model_service) -> None:
    """CA-15: not-found CommError is 404, other CommError is 503."""
    model_service.get_predictor.side_effect = wandb.errors.CommError(
        "artifact Not Found"
    )
    model_service.list_available_models.side_effect = (
        wandb.errors.CommError("timeout")
    )

    not_found_response = client.get("/models/default")
    unavailable_response = client.get("/models/available")

    assert not_found_response.status_code == 404
    assert unavailable_response.status_code == 503


def test_lookup_error_is_404(client, model_service) -> None:
    """CA-16: LookupError responds 404 with its message."""
    model_service.get_predictor.side_effect = LookupError("No servable")

    response = client.get("/models/default")

    assert response.status_code == 404
    assert response.json() == {"detail": "No servable"}


def test_openapi_lists_documented_routes(client) -> None:
    """CA-17: the schema has the 8 routes, each with a summary."""
    openapi_schema = client.get("/openapi.json").json()

    route_operations = {
        (route_path, method)
        for route_path, path_item in openapi_schema["paths"].items()
        for method in path_item
    }
    assert route_operations == {
        ("/health", "get"),
        ("/models/loaded", "get"),
        ("/models/available", "get"),
        ("/models/default", "get"),
        ("/models/reload", "post"),
        ("/predictions/log-returns", "post"),
        ("/predictions/prices", "post"),
        ("/predictions/market", "get"),
    }
    assert all(
        "summary" in operation
        for path_item in openapi_schema["paths"].values()
        for operation in path_item.values()
    )
