"""Unit tests for the FastAPI application."""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from wandb.errors import CommError
from wandb.sdk.mailbox.mailbox import MailboxClosedError

from generative_models.serving.api import app, get_model_service, main
from generative_models.serving.model_service import AvailableModel

FETCH_PATH = "generative_models.serving.predict_autoencoder.fetch_price_series"
EXPECTED_PATHS = {
    "/health",
    "/models/loaded",
    "/models/available",
    "/models/default",
    "/models/reload",
    "/predictions/log-returns",
    "/predictions/prices",
    "/predictions/market",
}


@pytest.fixture(name="mock_service")
def _mock_service(monkeypatch, make_predictor):
    """Replace the model service with a mock for every route test."""
    monkeypatch.setenv("AUTOENCODER_PRELOAD_MODEL", "false")
    predictor = make_predictor(version="v2")
    service = MagicMock()
    service.default_ticker = "PETR4.SA"
    service.get_predictor.return_value = predictor
    service.reload.return_value = predictor
    service.list_loaded_models.return_value = [predictor.metadata]
    service.list_available_models.return_value = [
        AvailableModel(
            artifact_name="simple-autoencoder-aapl:v4",
            ticker="AAPL",
            created_at="2026-09-20",
        ),
        AvailableModel(
            artifact_name="simple-autoencoder-petr4.sa:v2",
            ticker="PETR4.SA",
            created_at="2026-09-10",
        ),
    ]
    app.dependency_overrides[get_model_service] = lambda: service
    yield service
    app.dependency_overrides.clear()


@pytest.fixture(name="api_client")
def _api_client(mock_service):
    """Start the application with the mocked service in place."""
    assert app.dependency_overrides[get_model_service]() is mock_service
    with TestClient(app) as test_client:
        yield test_client


def test_startup_preloads_the_default_model(monkeypatch, mock_service) -> None:
    """Check that the default model is requested when the API starts."""
    monkeypatch.setenv("AUTOENCODER_PRELOAD_MODEL", "true")

    with TestClient(app):
        pass

    mock_service.get_predictor.assert_any_call()


def test_startup_skips_preload_when_disabled(mock_service) -> None:
    """Check that AUTOENCODER_PRELOAD_MODEL=false skips the preload."""
    with TestClient(app):
        pass

    mock_service.get_predictor.assert_not_called()


def test_startup_failure_does_not_stop_the_api(
    monkeypatch, mock_service
) -> None:
    """Check that a failed preload is logged and the API still answers."""
    monkeypatch.setenv("AUTOENCODER_PRELOAD_MODEL", "true")
    mock_service.get_predictor.side_effect = RuntimeError("no internet")

    with TestClient(app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200


def test_health_reports_status_and_loaded_tickers(api_client) -> None:
    """Check the exact body of the health route."""
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "default_ticker": "PETR4.SA",
        "loaded_tickers": ["PETR4.SA"],
    }


def test_loaded_models_route_returns_in_memory_metadata(api_client) -> None:
    """Check the metadata returned for the models in memory."""
    response = api_client.get("/models/loaded")

    assert response.status_code == 200
    artifact_name = response.json()[0]["artifact_name"]
    assert artifact_name == "simple-autoencoder-petr4.sa:v2"


def test_available_models_route_keeps_service_order(api_client) -> None:
    """Check that the available models keep the service's order."""
    response = api_client.get("/models/available")

    assert response.status_code == 200
    assert [model["ticker"] for model in response.json()] == [
        "AAPL",
        "PETR4.SA",
    ]


def test_default_model_route_returns_default_metadata(
    api_client, mock_service
) -> None:
    """Check that the default route returns the default model metadata."""
    response = api_client.get("/models/default")

    assert response.status_code == 200
    assert response.json()["ticker"] == "PETR4.SA"
    mock_service.get_predictor.assert_called_with()


def test_reload_route_passes_the_ticker_to_the_service(
    api_client, mock_service
) -> None:
    """Check that reload receives the ticker, or None when omitted."""
    named_response = api_client.post("/models/reload", json={"ticker": "AAPL"})
    unnamed_response = api_client.post("/models/reload", json={})

    assert named_response.status_code == 200
    assert unnamed_response.status_code == 200
    mock_service.reload.assert_any_call("AAPL")
    mock_service.reload.assert_any_call(None)


def test_log_return_prediction_returns_eight_windows(
    api_client,
    mock_service,
) -> None:
    """Check the windows of a prediction for 12 log-returns."""
    log_returns = np.linspace(-0.02, 0.02, 12).tolist()

    response = api_client.post(
        "/predictions/log-returns",
        json={"ticker": "PETR4.SA", "log_returns": log_returns},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["number_of_windows"] == 8
    assert [window["window_end_index"] for window in body["windows"]] == (
        list(range(4, 12))
    )
    assert all(window["window_end_date"] is None for window in body["windows"])
    mock_service.get_predictor.assert_called_with("PETR4.SA")


def test_log_return_prediction_with_too_few_values_returns_422(
    api_client,
) -> None:
    """Check that fewer log-returns than the window size give HTTP 422."""
    response = api_client.post(
        "/predictions/log-returns",
        json={"log_returns": [0.01, 0.02, 0.03, 0.04]},
    )

    assert response.status_code == 422
    assert "At least 5 log-returns are required" in response.json()["detail"]


def test_price_prediction_returns_seven_windows(api_client) -> None:
    """Check the windows of a prediction for 12 closing prices."""
    prices = (100 * np.exp(np.cumsum(np.full(12, 0.01)))).tolist()

    response = api_client.post("/predictions/prices", json={"prices": prices})

    assert response.status_code == 200
    assert response.json()["number_of_windows"] == 7


def test_price_prediction_rejects_zero_prices(api_client) -> None:
    """Check that a zero price is rejected by the request schema."""
    response = api_client.post(
        "/predictions/prices",
        json={"prices": [10.0, 0.0, 11.0, 12.0, 13.0, 14.0]},
    )

    assert response.status_code == 422


def test_market_route_downloads_prices_and_fills_dates(
    api_client,
    make_price_frame,
) -> None:
    """Check the market route with 10 prices from 2026-09-01."""
    price_data = make_price_frame(10)

    with patch(FETCH_PATH, return_value=price_data) as mock_fetch:
        response = api_client.get(
            "/predictions/market",
            params={"start_date": "2026-09-01", "end_date": "2026-09-30"},
        )

    body = response.json()
    assert response.status_code == 200
    mock_fetch.assert_called_once_with("PETR4.SA", "2026-09-01", "2026-09-30")
    assert body["number_of_windows"] == 5
    assert body["windows"][0]["window_end_index"] == 4
    assert body["windows"][-1]["window_end_date"] == "2026-09-14"
    assert body["windows"][-1]["close_price"] == pytest.approx(
        float(price_data["close_price"].iloc[-1])
    )


def test_market_route_rejects_reversed_dates(api_client) -> None:
    """Check that start_date must be earlier than end_date."""
    response = api_client.get(
        "/predictions/market",
        params={"start_date": "2026-10-01", "end_date": "2026-09-01"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "start_date must be earlier than end_date."
    )


def test_market_route_defaults_to_last_365_days(
    api_client,
    make_price_frame,
) -> None:
    """Check the default period when no dates are given."""
    with patch(FETCH_PATH, return_value=make_price_frame(10)) as mock_fetch:
        api_client.get("/predictions/market")

    start_argument, end_argument = mock_fetch.call_args.args[1:]
    assert end_argument == date.today().isoformat()
    assert start_argument < end_argument


def test_wandb_not_found_error_returns_404(api_client, mock_service) -> None:
    """Check that a missing wandb artifact gives HTTP 404."""
    mock_service.get_predictor.side_effect = CommError("artifact not found")

    response = api_client.get("/models/default")

    assert response.status_code == 404
    assert response.json()["detail"].startswith("Model artifact not found")


def test_wandb_other_error_returns_503(api_client, mock_service) -> None:
    """Check that other wandb communication errors give HTTP 503."""
    mock_service.list_available_models.side_effect = CommError("timeout")

    response = api_client.get("/models/available")

    assert response.status_code == 503
    assert response.json()["detail"].startswith("wandb is unavailable")


def test_lookup_error_returns_404_with_its_message(
    api_client,
    mock_service,
) -> None:
    """Check that a LookupError becomes HTTP 404 with the bare message."""
    mock_service.get_predictor.side_effect = LookupError("No servable")

    response = api_client.get("/models/default")

    assert response.status_code == 404
    assert response.json() == {"detail": "No servable"}


def test_key_error_message_has_no_extra_quotes(
    api_client,
    mock_service,
) -> None:
    """Check that a KeyError message is returned without Python's quotes."""
    mock_service.get_predictor.side_effect = KeyError(
        "No model loaded for ticker 'AAPL'."
    )

    response = api_client.get("/models/default")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "No model loaded for ticker 'AAPL'.",
    }


def test_openapi_lists_exactly_the_documented_routes(api_client) -> None:
    """Check the paths of the OpenAPI document and their summaries."""
    schema = api_client.get("/openapi.json").json()

    assert set(schema["paths"].keys()) == EXPECTED_PATHS
    for path_item in schema["paths"].values():
        for operation in path_item.values():
            assert "summary" in operation


@patch("generative_models.serving.api.uvicorn.run")
def test_main_starts_uvicorn_with_host_and_port(mock_uvicorn_run) -> None:
    """Check that main passes the host and port to uvicorn."""
    main(host="0.0.0.0", port=9000)

    mock_uvicorn_run.assert_called_once_with(app, host="0.0.0.0", port=9000)


@patch("generative_models.serving.api.AutoencoderModelService")
def test_closed_wandb_connection_returns_503_and_resets_client(
    mock_service_class,
    api_client,
    mock_service,
) -> None:
    """Check that a closed wandb connection gives 503 and a client reset."""
    mock_service.list_available_models.side_effect = MailboxClosedError()

    response = api_client.get("/models/available")

    assert response.status_code == 503
    assert "retry the request" in response.json()["detail"]
    registry = mock_service_class.get_instance.return_value.registry
    registry.reset_api.assert_called_once_with()
