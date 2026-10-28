"""Unit tests for the autoencoder REST API."""

# pylint: disable=duplicate-code

from typing import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import wandb
from fastapi.testclient import TestClient

from generative_models.models.autoencoder.autoencoder import (
    SimpleAutoencoder,
)
from generative_models.serving.api import app, get_model_service
from generative_models.serving.autoencoder_predictor import (
    AutoencoderModelMetadata,
    AutoencoderPredictor,
)
from generative_models.serving.model_service import AvailableModel

PREDICT_MODULE = "generative_models.serving.predict_autoencoder"
WINDOW_SIZE = 5


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
        model=SimpleAutoencoder(input_dim=WINDOW_SIZE),
        metadata=AutoencoderModelMetadata(
            ticker=ticker,
            window_size=WINDOW_SIZE,
            return_train_mean=0.001,
            return_train_std=0.02,
            anomaly_threshold=0.5,
            artifact_name=f"simple-autoencoder-{ticker.lower()}:v1",
        ),
        device="cpu",
    )


def build_service_mock() -> MagicMock:
    """Build a model service serving one in-memory predictor.

    Returns
    -------
    MagicMock
        Stand-in service whose default model is PETR4.SA.
    """
    predictor = build_predictor()
    service = MagicMock()
    service.default_ticker = "PETR4.SA"
    service.get_predictor.return_value = predictor
    service.reload.return_value = predictor
    service.list_loaded_models.return_value = [predictor.metadata]
    service.list_available_models.return_value = [
        AvailableModel(
            artifact_name="simple-autoencoder-petr4.sa:v1",
            ticker="PETR4.SA",
            created_at="2026-05-01T00:00:00",
        ),
    ]

    return service


@pytest.fixture(name="service")
def service_fixture() -> Iterator[MagicMock]:
    """Install a mocked model service for the duration of a test."""
    service = build_service_mock()
    app.dependency_overrides[get_model_service] = lambda: service
    yield service
    app.dependency_overrides.clear()


def build_log_returns(number_of_returns: int) -> list[float]:
    """Build a synthetic log-return series as plain floats.

    Parameters
    ----------
    number_of_returns : int
        Number of log-returns to generate.

    Returns
    -------
    list[float]
        Values ready to be sent as JSON.
    """
    random_generator = np.random.default_rng(13)

    return random_generator.normal(
        loc=0.0,
        scale=0.02,
        size=number_of_returns,
    ).tolist()


def build_price_dataframe(number_of_prices: int) -> pd.DataFrame:
    """Build a synthetic price series indexed by business day.

    Parameters
    ----------
    number_of_prices : int
        Number of daily prices to generate.

    Returns
    -------
    pd.DataFrame
        Single-column data frame holding the prices in close_price.
    """
    random_generator = np.random.default_rng(5)
    log_increments = random_generator.normal(
        loc=0.0,
        scale=0.02,
        size=number_of_prices,
    )
    price_values = 100.0 * np.exp(np.cumsum(log_increments))
    price_dates = pd.bdate_range("2026-09-01", periods=number_of_prices)

    return pd.DataFrame({"close_price": price_values}, index=price_dates)


def test_startup_preloads_the_default_model(service: MagicMock) -> None:
    """CA-01: starting the application loads the default model."""
    with TestClient(app):
        pass

    service.get_predictor.assert_called_once_with()


def test_preload_can_be_disabled(
    service: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CA-02: AUTOENCODER_PRELOAD_MODEL=false skips the preload."""
    monkeypatch.setenv("AUTOENCODER_PRELOAD_MODEL", "false")

    with TestClient(app):
        pass

    service.get_predictor.assert_not_called()


def test_startup_survives_a_preload_failure(service: MagicMock) -> None:
    """CA-03: a failing preload does not prevent the API from serving."""
    service.get_predictor.side_effect = RuntimeError("no internet")

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200


def test_health_reports_the_loaded_models(service: MagicMock) -> None:
    """CA-04: health returns the status, default ticker and tickers."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "default_ticker": "PETR4.SA",
        "loaded_tickers": ["PETR4.SA"],
    }
    service.list_available_models.assert_not_called()


@pytest.mark.usefixtures("service")
def test_loaded_models_route() -> None:
    """CA-05: the loaded models route returns their metadata."""
    with TestClient(app) as client:
        response = client.get("/models/loaded")

    payload = response.json()

    assert response.status_code == 200
    assert len(payload) == 1
    assert payload[0]["ticker"] == "PETR4.SA"
    assert payload[0]["artifact_name"] == (
        "simple-autoencoder-petr4.sa:v1"
    )
    assert payload[0]["window_size"] == WINDOW_SIZE


def test_available_models_route(service: MagicMock) -> None:
    """CA-06: the available models route keeps the service order."""
    with TestClient(app) as client:
        response = client.get("/models/available")

    assert response.status_code == 200
    assert response.json() == [
        {
            "artifact_name": "simple-autoencoder-petr4.sa:v1",
            "ticker": "PETR4.SA",
            "created_at": "2026-05-01T00:00:00",
        }
    ]
    service.list_available_models.assert_called_once()


def test_default_model_route(service: MagicMock) -> None:
    """CA-07: the default model route describes the served model."""
    with TestClient(app) as client:
        response = client.get("/models/default")

    assert response.status_code == 200
    assert response.json()["ticker"] == "PETR4.SA"
    assert service.get_predictor.called


def test_reload_route_forwards_the_ticker(service: MagicMock) -> None:
    """CA-08: reload passes the ticker through, or None when absent."""
    with TestClient(app) as client:
        with_ticker = client.post(
            "/models/reload",
            json={"ticker": "AAPL"},
        )
        without_ticker = client.post("/models/reload", json={})

    assert with_ticker.status_code == 200
    assert without_ticker.status_code == 200
    assert [call.args[0] for call in service.reload.call_args_list] == [
        "AAPL",
        None,
    ]


def test_log_return_prediction_route(service: MagicMock) -> None:
    """CA-09: twelve log-returns produce eight undated windows."""
    with TestClient(app) as client:
        response = client.post(
            "/predictions/log-returns",
            json={"log_returns": build_log_returns(12)},
        )
        with_ticker = client.post(
            "/predictions/log-returns",
            json={"ticker": "AAPL", "log_returns": build_log_returns(12)},
        )

    payload = response.json()

    assert response.status_code == 200
    assert payload["number_of_windows"] == 8
    assert [window["window_end_index"] for window in payload["windows"]] == (
        list(range(4, 12))
    )
    assert all(
        window["window_end_date"] is None for window in payload["windows"]
    )
    assert with_ticker.status_code == 200
    assert service.get_predictor.call_args.args[0] == "AAPL"


@pytest.mark.usefixtures("service")
def test_log_return_prediction_rejects_short_series() -> None:
    """CA-10: fewer log-returns than the window answers 422."""
    with TestClient(app) as client:
        response = client.post(
            "/predictions/log-returns",
            json={"log_returns": build_log_returns(3)},
        )

    assert response.status_code == 422
    assert "At least 5 log-returns" in response.json()["detail"]


@pytest.mark.usefixtures("service")
def test_price_prediction_route() -> None:
    """CA-11: twelve prices produce seven windows."""
    price_values = (
        100.0 * np.exp(np.cumsum(build_log_returns(12)))
    ).tolist()

    with TestClient(app) as client:
        response = client.post(
            "/predictions/prices",
            json={"prices": price_values},
        )

    assert response.status_code == 200
    assert response.json()["number_of_windows"] == 7


@pytest.mark.usefixtures("service")
def test_price_prediction_rejects_non_positive_prices() -> None:
    """CA-12: a zero price is rejected by the schema with 422."""
    with TestClient(app) as client:
        response = client.post(
            "/predictions/prices",
            json={"prices": [10.0, 0.0, 11.0, 12.0, 13.0, 14.0]},
        )

    assert response.status_code == 422


@pytest.mark.usefixtures("service")
def test_market_prediction_route() -> None:
    """CA-13: market data is fetched for the period and dated."""
    price_data = build_price_dataframe(10)

    with patch(
        f"{PREDICT_MODULE}.fetch_price_series",
        return_value=price_data,
    ) as fetch_mock:
        with TestClient(app) as client:
            response = client.get(
                "/predictions/market",
                params={
                    "start_date": "2026-09-01",
                    "end_date": "2026-09-30",
                },
            )

    payload = response.json()

    fetch_mock.assert_called_once_with(
        "PETR4.SA",
        "2026-09-01",
        "2026-09-30",
    )
    assert response.status_code == 200
    assert payload["number_of_windows"] == 5
    assert payload["windows"][0]["window_end_index"] == 4
    assert payload["windows"][-1]["window_end_date"] == "2026-09-14"
    assert payload["windows"][-1]["close_price"] == pytest.approx(
        float(price_data["close_price"].iloc[-1])
    )


@pytest.mark.usefixtures("service")
def test_market_prediction_rejects_inverted_dates() -> None:
    """CA-14: a start date after the end date answers 422."""
    with TestClient(app) as client:
        response = client.get(
            "/predictions/market",
            params={
                "start_date": "2026-09-30",
                "end_date": "2026-09-01",
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "start_date must be earlier than end_date."
    )


def test_wandb_errors_become_404_and_503(service: MagicMock) -> None:
    """CA-15: a missing artifact is 404 and an unreachable wandb is 503."""
    service.get_predictor.side_effect = wandb.errors.CommError(
        "artifact not found"
    )
    service.list_available_models.side_effect = wandb.errors.CommError(
        "timeout"
    )

    with TestClient(app) as client:
        missing_response = client.get("/models/default")
        unavailable_response = client.get("/models/available")

    assert missing_response.status_code == 404
    assert "Model artifact not found" in missing_response.json()["detail"]
    assert unavailable_response.status_code == 503
    assert "wandb is unavailable" in unavailable_response.json()["detail"]


def test_lookup_errors_become_404(service: MagicMock) -> None:
    """CA-16: a LookupError answers 404 with its own message."""
    service.get_predictor.side_effect = LookupError("No servable")

    with TestClient(app) as client:
        response = client.get("/models/default")

    assert response.status_code == 404
    assert response.json() == {"detail": "No servable"}


@pytest.mark.usefixtures("service")
def test_openapi_documents_every_route() -> None:
    """CA-17: the schema holds the eight routes, all with a summary."""
    with TestClient(app) as client:
        openapi_schema = client.get("/openapi.json").json()

    assert set(openapi_schema["paths"]) == {
        "/health",
        "/models/loaded",
        "/models/available",
        "/models/default",
        "/models/reload",
        "/predictions/log-returns",
        "/predictions/prices",
        "/predictions/market",
    }
    assert all(
        "summary" in operation
        for path_item in openapi_schema["paths"].values()
        for operation in path_item.values()
    )
