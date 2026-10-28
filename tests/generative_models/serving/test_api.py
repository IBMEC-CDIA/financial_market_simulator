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

FETCH_PRICE_SERIES_PATH = (
    "generative_models.serving.predict_autoencoder.fetch_price_series"
)


def build_predictor() -> AutoencoderPredictor:
    """Create a predictor around an untrained autoencoder.

    Returns
    -------
    AutoencoderPredictor
        Predictor of `PETR4.SA` with `window_size=5`.
    """
    torch.manual_seed(0)
    return AutoencoderPredictor(
        model=SimpleAutoencoder(input_dim=5),
        metadata=AutoencoderModelMetadata(
            ticker="PETR4.SA",
            window_size=5,
            return_train_mean=0.0,
            return_train_std=0.02,
            anomaly_threshold=1.0,
            artifact_name="simple-autoencoder-petr4.sa:v1",
        ),
    )


def build_prices(number_of_prices: int) -> np.ndarray:
    """Create a reproducible synthetic positive price series.

    Parameters
    ----------
    number_of_prices : int
        Number of prices.

    Returns
    -------
    np.ndarray
        Positive prices around 100.
    """
    random_generator = np.random.default_rng(5)
    log_returns = random_generator.normal(0.0, 0.02, number_of_prices)
    return 100.0 * np.exp(np.cumsum(log_returns))


@pytest.fixture(name="predictor")
def fixture_predictor() -> AutoencoderPredictor:
    """Provide the predictor served by the fake model service.

    Returns
    -------
    AutoencoderPredictor
        Predictor of `PETR4.SA`.
    """
    return build_predictor()


@pytest.fixture(name="model_service")
def fixture_model_service(
    predictor: AutoencoderPredictor,
) -> Iterator[MagicMock]:
    """Override the model service dependency with a mock.

    Parameters
    ----------
    predictor : AutoencoderPredictor
        Predictor returned by `get_predictor` and `reload`.

    Returns
    -------
    Iterator[MagicMock]
        Generator yielding the mock and clearing the overrides.
    """
    fake_model_service = MagicMock()
    fake_model_service.default_ticker = "PETR4.SA"
    fake_model_service.get_predictor.return_value = predictor
    fake_model_service.reload.return_value = predictor
    fake_model_service.list_loaded_models.return_value = [
        predictor.metadata
    ]
    app.dependency_overrides[get_model_service] = (
        lambda: fake_model_service
    )
    yield fake_model_service
    app.dependency_overrides.clear()


@pytest.fixture(name="client")
def fixture_client(model_service: MagicMock) -> Iterator[TestClient]:
    """Start the API with the fake model service.

    Parameters
    ----------
    model_service : MagicMock
        Fake model service already installed as an override.

    Returns
    -------
    Iterator[TestClient]
        Generator yielding a client whose lifespan has run.
    """
    with TestClient(app) as test_client:
        model_service.get_predictor.reset_mock()
        yield test_client


def expected_metadata(predictor: AutoencoderPredictor) -> dict:
    """Build the expected JSON of a model metadata response.

    Parameters
    ----------
    predictor : AutoencoderPredictor
        Predictor whose metadata is expected.

    Returns
    -------
    dict
        Metadata fields as returned by the API.
    """
    metadata = predictor.metadata
    return {
        "ticker": metadata.ticker,
        "artifact_name": metadata.artifact_name,
        "window_size": metadata.window_size,
        "return_train_mean": metadata.return_train_mean,
        "return_train_std": metadata.return_train_std,
        "anomaly_threshold": metadata.anomaly_threshold,
    }


def test_startup_preloads_default_model(model_service: MagicMock) -> None:
    """CA-01: check that startup loads the default model."""
    with TestClient(app):
        pass

    model_service.get_predictor.assert_called_once_with()


def test_startup_can_skip_preload(
    model_service: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CA-02: check that AUTOENCODER_PRELOAD_MODEL=false skips loading."""
    monkeypatch.setenv("AUTOENCODER_PRELOAD_MODEL", "FALSE")

    with TestClient(app):
        pass

    model_service.get_predictor.assert_not_called()


def test_startup_survives_loading_failure(model_service: MagicMock) -> None:
    """CA-03: check that a preload failure does not stop the API."""
    model_service.get_predictor.side_effect = RuntimeError("offline")

    with TestClient(app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200


def test_health_reports_loaded_models(client: TestClient) -> None:
    """CA-04: check the exact health response."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "default_ticker": "PETR4.SA",
        "loaded_tickers": ["PETR4.SA"],
    }


def test_loaded_models_route(
    client: TestClient,
    predictor: AutoencoderPredictor,
) -> None:
    """CA-05: check the metadata of the models in memory."""
    response = client.get("/models/loaded")

    assert response.status_code == 200
    assert response.json() == [expected_metadata(predictor)]


def test_available_models_route(
    client: TestClient,
    model_service: MagicMock,
) -> None:
    """CA-06: check that the available models keep the service order."""
    model_service.list_available_models.return_value = [
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

    response = client.get("/models/available")

    assert response.status_code == 200
    assert [model["ticker"] for model in response.json()] == [
        "PETR4.SA",
        "AAPL",
    ]
    assert response.json()[0]["artifact_name"] == (
        "simple-autoencoder-petr4.sa:v4"
    )


def test_default_model_route(
    client: TestClient,
    predictor: AutoencoderPredictor,
    model_service: MagicMock,
) -> None:
    """CA-07: check the metadata of the default model."""
    response = client.get("/models/default")

    assert response.status_code == 200
    assert response.json() == expected_metadata(predictor)
    model_service.get_predictor.assert_called_once_with()


def test_reload_route_forwards_ticker(
    client: TestClient,
    model_service: MagicMock,
) -> None:
    """CA-08: check the ticker passed to reload."""
    ticker_response = client.post("/models/reload", json={"ticker": "AAPL"})
    default_response = client.post("/models/reload", json={})

    assert ticker_response.status_code == 200
    assert default_response.status_code == 200
    assert [call.args for call in model_service.reload.call_args_list] == [
        ("AAPL",),
        (None,),
    ]


def test_log_return_prediction(
    client: TestClient,
    model_service: MagicMock,
) -> None:
    """CA-09: check windows, indices and the requested ticker."""
    log_returns = np.diff(np.log(build_prices(13))).tolist()

    response = client.post(
        "/predictions/log-returns",
        json={"ticker": "PETR4.SA", "log_returns": log_returns},
    )

    assert response.status_code == 200
    response_body = response.json()
    assert response_body["number_of_windows"] == 8
    assert [
        window["window_end_index"] for window in response_body["windows"]
    ] == list(range(4, 12))
    assert all(
        window["window_end_date"] is None
        for window in response_body["windows"]
    )
    assert response_body["outlier_rate"] == (
        response_body["number_of_outliers"] / 8
    )
    model_service.get_predictor.assert_called_once_with("PETR4.SA")


def test_log_return_prediction_with_short_series(client: TestClient) -> None:
    """CA-10: check that fewer values than window_size answer 422."""
    response = client.post(
        "/predictions/log-returns",
        json={"log_returns": [0.01, 0.02]},
    )

    assert response.status_code == 422
    assert "At least 5 log-returns" in response.json()["detail"]


def test_price_prediction(client: TestClient) -> None:
    """CA-11: check the number of windows of 12 prices."""
    response = client.post(
        "/predictions/prices",
        json={"prices": build_prices(12).tolist()},
    )

    assert response.status_code == 200
    assert response.json()["number_of_windows"] == 7


def test_price_prediction_rejects_zero_price(client: TestClient) -> None:
    """CA-12: check that a zero price answers 422."""
    response = client.post(
        "/predictions/prices",
        json={"prices": [10.0, 0.0, 11.0, 12.0, 13.0, 14.0, 15.0]},
    )

    assert response.status_code == 422


def test_market_prediction(client: TestClient) -> None:
    """CA-13: check the downloaded period and the window dates."""
    price_data = pd.DataFrame(
        {"close_price": build_prices(10)},
        index=pd.bdate_range("2026-09-01", periods=10),
    )

    with patch(
        FETCH_PRICE_SERIES_PATH,
        return_value=price_data,
    ) as mock_fetch_price_series:
        response = client.get(
            "/predictions/market",
            params={"start_date": "2026-09-01", "end_date": "2026-09-30"},
        )

    assert response.status_code == 200
    mock_fetch_price_series.assert_called_once_with(
        "PETR4.SA",
        "2026-09-01",
        "2026-09-30",
    )
    windows = response.json()["windows"]
    assert len(windows) == 5
    assert windows[0]["window_end_index"] == 4
    assert windows[-1]["window_end_date"] == "2026-09-14"
    assert windows[-1]["close_price"] == pytest.approx(
        price_data["close_price"].iloc[-1]
    )


def test_market_prediction_rejects_inverted_period(
    client: TestClient,
) -> None:
    """CA-14: check that start_date after end_date answers 422."""
    response = client.get(
        "/predictions/market",
        params={"start_date": "2026-09-30", "end_date": "2026-09-01"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "start_date must be earlier than end_date."
    }


def test_wandb_errors_are_translated(
    client: TestClient,
    model_service: MagicMock,
) -> None:
    """CA-15: check the 404 and 503 answers of wandb errors."""
    model_service.get_predictor.side_effect = wandb.errors.CommError(
        "artifact Not Found"
    )
    model_service.list_available_models.side_effect = (
        wandb.errors.CommError("timeout")
    )

    not_found_response = client.get("/models/default")
    unavailable_response = client.get("/models/available")

    assert not_found_response.status_code == 404
    assert not_found_response.json()["detail"].startswith(
        "Model artifact not found:"
    )
    assert unavailable_response.status_code == 503
    assert unavailable_response.json()["detail"] == (
        "wandb is unavailable: timeout"
    )


def test_lookup_error_answers_404(
    client: TestClient,
    model_service: MagicMock,
) -> None:
    """CA-16: check the 404 answer of a LookupError."""
    model_service.get_predictor.side_effect = LookupError("No servable")

    response = client.get("/models/default")

    assert response.status_code == 404
    assert response.json() == {"detail": "No servable"}


def test_key_error_detail_has_no_quotes(
    client: TestClient,
    model_service: MagicMock,
) -> None:
    """Check that the KeyError quotes are removed from the detail."""
    model_service.get_predictor.side_effect = KeyError("No model loaded")

    response = client.get("/models/default")

    assert response.status_code == 404
    assert response.json() == {"detail": "No model loaded"}


def test_openapi_documents_every_route(client: TestClient) -> None:
    """CA-17: check the documented routes and their summaries."""
    openapi_schema = client.get("/openapi.json").json()

    documented_routes = {
        (method.upper(), path)
        for path, operations in openapi_schema["paths"].items()
        for method in operations
    }

    assert documented_routes == {
        ("GET", "/health"),
        ("GET", "/models/loaded"),
        ("GET", "/models/available"),
        ("GET", "/models/default"),
        ("POST", "/models/reload"),
        ("POST", "/predictions/log-returns"),
        ("POST", "/predictions/prices"),
        ("GET", "/predictions/market"),
    }
    assert all(
        "summary" in operation
        for operations in openapi_schema["paths"].values()
        for operation in operations.values()
    )
