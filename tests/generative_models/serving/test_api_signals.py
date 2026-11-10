"""Integration tests of the GET /signals route."""

# pylint: disable=duplicate-code

from datetime import date, datetime

import pandas as pd
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
from generative_models.serving import api as serving_api
from generative_models.serving import signals as signal_rules
from generative_models.serving.api import app
from generative_models.serving.model_service import (
    AutoencoderModelService,
)
from generative_models.serving.signals import SignalDecision

WINDOW_SIZE = 5
THRESHOLD = 1.25


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
        anomaly_threshold=THRESHOLD,
    )
    save_overfitting_report(report, directory / f"{name}-overfitting.json")


def _fake_prediction_table(
    closes: list[float],
    scores: list[float],
) -> pd.DataFrame:
    """Build a prediction table like the one of the market data route."""
    dates = pd.date_range("2026-10-01", periods=len(closes), freq="B")

    return pd.DataFrame(
        {
            "close_price": closes,
            "reconstruction_score": scores,
            "is_outlier": [score > THRESHOLD for score in scores],
        },
        index=dates,
    )


@pytest.fixture(name="signals_client")
def _signals_client(monkeypatch, tmp_path):
    """Start the API with two models read from a temporary directory."""
    _write_model(tmp_path, "AAPL", "simple-autoencoder-aapl")
    _write_model(tmp_path, "MSFT", "simple-autoencoder-msft")
    monkeypatch.setenv("AUTOENCODER_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("AUTOENCODER_DEVICE", "cpu")
    AutoencoderModelService.reset_instance()
    with TestClient(app) as client:
        yield client
    AutoencoderModelService.reset_instance()


@pytest.fixture(name="fake_prices")
def _fake_prices(monkeypatch):
    """Replace the price download with fixed prediction tables.

    AAPL has an abnormal drop in its last window. MSFT has too few values,
    so its prediction raises a ValueError.

    Returns
    -------
    list[tuple[str, str]]
        Calls made to the fake prediction, as (start, end) dates.
    """
    calls = []

    def fake_predict_period(predictor, start_date, end_date):
        calls.append((start_date, end_date))
        if predictor.metadata.ticker == "MSFT":
            raise ValueError("At least 6 prices are required, got 3.")
        return _fake_prediction_table([10.0, 9.0], [0.5, 2.0])

    monkeypatch.setattr(serving_api, "predict_period", fake_predict_period)

    return calls


def test_signals_is_an_empty_list_without_a_rule(
    signals_client,
    fake_prices,
    monkeypatch,
) -> None:
    """Check that the route answers [] and fetches nothing without a rule."""
    monkeypatch.setattr(signal_rules, "default_signal_rule", None)

    response = signals_client.get("/signals", params={"ticker": "AAPL"})

    assert response.status_code == 200
    assert response.json() == []
    assert not fake_prices


def test_signals_requires_a_ticker(signals_client) -> None:
    """Check that a request without the ticker is refused."""
    response = signals_client.get("/signals")

    assert response.status_code == 422


def test_signals_uses_the_given_period(
    signals_client,
    fake_prices,
) -> None:
    """Check that the start and end dates are sent to the price download."""
    signals_client.get(
        "/signals",
        params={
            "ticker": "AAPL",
            "start_date": "2026-01-01",
            "end_date": "2026-06-30",
        },
    )

    assert fake_prices == [("2026-01-01", "2026-06-30")]


def test_signals_defaults_to_the_last_year(
    signals_client,
    fake_prices,
) -> None:
    """Check that a missing start date means 365 days before the end."""
    signals_client.get(
        "/signals",
        params={"ticker": "AAPL", "end_date": "2026-06-30"},
    )

    start_date, end_date = fake_prices[0]
    assert end_date == "2026-06-30"
    assert (
        date.fromisoformat(end_date) - date.fromisoformat(start_date)
    ).days == serving_api.DEFAULT_LOOKBACK_DAYS


@pytest.mark.usefixtures("fake_prices")
def test_signals_rejects_a_start_date_after_the_end_date(
    signals_client,
) -> None:
    """Check that an inverted period is refused without fetching prices."""
    response = signals_client.get(
        "/signals",
        params={
            "ticker": "AAPL",
            "start_date": "2026-06-30",
            "end_date": "2026-01-01",
        },
    )

    assert response.status_code == 422


@pytest.mark.usefixtures("fake_prices")
def test_signals_returns_the_item_of_the_ticker(
    signals_client,
) -> None:
    """Check the fields and values of the item of the requested model."""
    response = signals_client.get("/signals", params={"ticker": "aapl"})

    assert response.status_code == 200
    items = response.json()
    assert len(items) == 1
    item = items[0]
    assert set(item) == {
        "symbol",
        "action",
        "confidence",
        "generated_at",
        "model",
        "reason",
    }
    assert item["symbol"] == "AAPL"
    assert item["action"] == "buy"
    assert item["confidence"] == pytest.approx(0.6)
    assert item["model"] == "simple-autoencoder-aapl:local"
    assert item["reason"].startswith("Janela de 5 dias")
    parsed_time = datetime.fromisoformat(item["generated_at"])
    assert parsed_time.utcoffset().total_seconds() == 0


@pytest.mark.usefixtures("fake_prices")
def test_signals_is_empty_when_the_period_has_too_few_prices(
    signals_client,
) -> None:
    """Check that a ticker without enough data gets no item."""
    response = signals_client.get("/signals", params={"ticker": "MSFT"})

    assert response.status_code == 200
    assert response.json() == []


def test_signals_unknown_ticker_is_not_found(
    signals_client,
    fake_prices,
) -> None:
    """Check that a ticker without a model answers 404."""
    response = signals_client.get("/signals", params={"ticker": "TSLA"})

    assert response.status_code == 404
    assert not fake_prices


@pytest.mark.usefixtures("fake_prices")
def test_signals_returns_the_rule_decision_of_the_model(
    signals_client,
    monkeypatch,
) -> None:
    """Check that the values of the rule decision reach the response."""

    def rule(metadata, _prediction_table) -> SignalDecision | None:
        if metadata.ticker == "AAPL":
            return SignalDecision(
                action="sell",
                confidence=0.75,
                reason="Janela de 10 dias com erro acima do limite.",
            )
        return None

    monkeypatch.setattr(signal_rules, "default_signal_rule", rule)

    response = signals_client.get("/signals", params={"ticker": "AAPL"})

    assert response.status_code == 200
    items = response.json()
    assert [item["symbol"] for item in items] == ["AAPL"]
    assert items[0]["action"] == "sell"
    assert items[0]["confidence"] == pytest.approx(0.75)


def test_existing_routes_still_answer(signals_client) -> None:
    """Check that the models route is not affected by the new route."""
    response = signals_client.get("/models/loaded")

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_signal_item_schema_matches_the_contract(signals_client) -> None:
    """Check the OpenAPI schema of the signal item against the spec 08."""
    schema = signals_client.get("/openapi.json").json()["components"][
        "schemas"
    ]["SignalItem"]

    assert set(schema["properties"]) == {
        "symbol",
        "action",
        "confidence",
        "generated_at",
        "model",
        "reason",
    }
    assert set(schema["required"]) == {
        "symbol",
        "action",
        "confidence",
        "generated_at",
    }
    assert schema["properties"]["action"]["enum"] == [
        "buy",
        "sell",
        "hold",
    ]


def test_history_route_gives_the_decision_of_each_day(
    signals_client,
    fake_prices,
) -> None:
    """Check that the history route answers one item per decided day."""
    response = signals_client.get(
        "/signals/history",
        params={
            "ticker": "AAPL",
            "start_date": "2026-10-01",
            "end_date": "2026-10-02",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["symbol"] == "AAPL"
    assert body[0]["action"] == "buy"
    assert body[0]["generated_at"].startswith("2026-10-02")
    assert fake_prices == [("2026-10-01", "2026-10-02")]


def test_history_route_is_empty_when_the_period_is_too_short(
    signals_client,
    fake_prices,
) -> None:
    """Check that the history route answers [] when the model cannot score."""
    response = signals_client.get(
        "/signals/history",
        params={
            "ticker": "MSFT",
            "start_date": "2026-10-01",
            "end_date": "2026-10-02",
        },
    )

    assert response.status_code == 200
    assert response.json() == []
    assert fake_prices == [("2026-10-01", "2026-10-02")]
