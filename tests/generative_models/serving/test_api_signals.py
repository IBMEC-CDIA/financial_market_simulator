"""Integration tests of the GET /signals route."""

# pylint: disable=duplicate-code

from datetime import datetime

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
from generative_models.serving import signals as signal_rules
from generative_models.serving.api import app
from generative_models.serving.model_service import (
    AutoencoderModelService,
)
from generative_models.serving.signals import SignalDecision

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


def test_signals_is_an_empty_list_without_a_rule(signals_client) -> None:
    """Check that the route answers [] while no rule is defined."""
    response = signals_client.get("/signals")

    assert response.status_code == 200
    assert response.json() == []


def test_signals_returns_one_item_per_model_with_a_decision(
    signals_client,
    monkeypatch,
) -> None:
    """Check the fields and the values of the items with a test rule."""

    def rule(metadata) -> SignalDecision | None:
        if metadata.ticker == "AAPL":
            return SignalDecision(
                action="sell",
                confidence=0.75,
                reason="Janela de 10 dias com erro acima do limite.",
            )
        return None

    monkeypatch.setattr(signal_rules, "DEFAULT_SIGNAL_RULE", rule)

    response = signals_client.get("/signals")

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
    assert item["action"] == "sell"
    assert item["confidence"] == pytest.approx(0.75)
    assert item["model"] == "simple-autoencoder-aapl:local"
    assert item["generated_at"].endswith("Z") or item[
        "generated_at"
    ].endswith("+00:00")
    parsed_time = datetime.fromisoformat(item["generated_at"])
    assert parsed_time.utcoffset().total_seconds() == 0


def test_existing_routes_still_answer(signals_client) -> None:
    """Check that the models route is not affected by the new route."""
    response = signals_client.get("/models/loaded")

    assert response.status_code == 200
    assert len(response.json()) == 2
