"""Tests of the GET /forecast route with a stored artifact and fake prices."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from generative_models.serving import api as serving_api
from generative_models.serving.api import app
from generative_models.serving.train_regression_forecaster import (
    TrainingConfig,
    save_regression_artifact,
    train_regression_forecaster,
)

AS_OF = date(2026, 5, 12)


def fake_prices(spike_after: date | None = None) -> pd.DataFrame:
    """Daily closes ending on AS_OF, optionally spiking after a day."""
    generator = np.random.default_rng(1)
    values = 100.0 * np.exp(np.cumsum(generator.normal(0.0005, 0.01, 200)))
    index = pd.bdate_range(end=pd.Timestamp(AS_OF), periods=200)
    if spike_after is not None:
        values = values.copy()
        values[index > pd.Timestamp(spike_after)] = 1_000_000.0
    return pd.DataFrame({"close_price": values}, index=index)


@pytest.fixture(name="client_with_artifact")
def _client_with_artifact(tmp_path, monkeypatch):
    """Client of the app with a trained artifact on disk."""
    closes = fake_prices()["close_price"]
    config = TrainingConfig(window_size=5, hidden_layer_size=4, epochs=2)
    result = train_regression_forecaster(closes, config)
    save_regression_artifact(result, "AAPL", tmp_path, config)
    monkeypatch.setenv("AUTOENCODER_MODEL_DIR", str(tmp_path))
    return TestClient(app, raise_server_exceptions=False)


def test_the_route_answers_with_the_forecast_contract(
    client_with_artifact,
    monkeypatch,
):
    """Client of the app with a trained artifact on disk."""
    monkeypatch.setattr(
        serving_api,
        "fetch_price_series",
        lambda *args: fake_prices(),
    )
    response = client_with_artifact.get(
        "/forecast",
        params={"ticker": "aapl", "as_of": AS_OF.isoformat()},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "ticker",
        "as_of",
        "target_date",
        "predicted_close",
        "window_days",
    }
    assert body["ticker"] == "AAPL"
    assert body["as_of"] == AS_OF.isoformat()
    assert body["window_days"] == 5
    assert body["predicted_close"] > 0


def test_a_ticker_without_artifact_is_a_404(client_with_artifact):
    """Check that a ticker without artifact is a 404."""
    response = client_with_artifact.get(
        "/forecast",
        params={"ticker": "TSLA", "as_of": AS_OF.isoformat()},
    )
    assert response.status_code == 404
    assert response.json() == {
        "detail": "No forecast model for ticker 'TSLA'."
    }


def test_a_future_as_of_is_a_422(client_with_artifact):
    """Check that a future as of is a 422."""
    future = (date.today() + timedelta(days=5)).isoformat()
    response = client_with_artifact.get(
        "/forecast",
        params={"ticker": "AAPL", "as_of": future},
    )
    assert response.status_code == 422
    assert "future" in response.json()["detail"]


def test_prices_after_as_of_do_not_change_the_answer(
    client_with_artifact,
    monkeypatch,
):
    """Check that prices after as of do not change the answer."""
    monkeypatch.setattr(
        serving_api,
        "fetch_price_series",
        lambda *args: fake_prices(),
    )
    clean = client_with_artifact.get(
        "/forecast",
        params={"ticker": "AAPL", "as_of": AS_OF.isoformat()},
    ).json()
    monkeypatch.setattr(
        serving_api,
        "fetch_price_series",
        lambda *args: fake_prices(spike_after=AS_OF),
    )
    with_future = client_with_artifact.get(
        "/forecast",
        params={"ticker": "AAPL", "as_of": AS_OF.isoformat()},
    ).json()
    assert clean == with_future


def test_the_price_request_covers_the_lookback_up_to_as_of(
    client_with_artifact,
    monkeypatch,
):
    """Check that the price request covers the lookback up to as of."""
    calls = []

    def recording_fetch(ticker, start, end):
        calls.append((ticker, start, end))
        return fake_prices()

    monkeypatch.setattr(serving_api, "fetch_price_series", recording_fetch)
    client_with_artifact.get(
        "/forecast",
        params={"ticker": "AAPL", "as_of": AS_OF.isoformat()},
    )
    ticker, start, end = calls[0]
    assert ticker == "AAPL"
    assert end == (AS_OF + timedelta(days=1)).isoformat()
    assert (AS_OF - date.fromisoformat(start)).days == 120
