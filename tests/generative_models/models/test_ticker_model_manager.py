"""Unit tests for TickerModelManager."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from generative_models.models.base_autoencoder.trainer_factory import (
    build_trainer,
)
from generative_models.models.ticker_model_manager import TickerModelManager

BUILD_DATASET_PATH = (
    "generative_models.models.ticker_model_manager.build_price_dataset"
)


def _build_manager(trainer_factory=None) -> TickerModelManager:
    """Create a manager with two tickers and an optional trainer factory."""
    return TickerModelManager(
        tickers={
            "AAPL": {"start_date": "2020-01-01", "end_date": "2026-01-01"},
            "MSFT": {"start_date": "2021-01-01", "end_date": "2026-01-01"},
        },
        trainer_factory=trainer_factory,
    )


def _make_factory() -> MagicMock:
    """Create a trainer factory whose trainers return a fake result."""
    factory = MagicMock()
    factory.return_value.train.return_value = MagicMock(name="result")

    return factory


def test_constructor_stores_tickers_configuration() -> None:
    """Check that the tickers dictionary is stored unchanged."""
    tickers = {
        "AAPL": {"start_date": "2020-01-01", "end_date": "2026-01-01"},
    }

    manager = TickerModelManager(tickers=tickers)

    assert manager.tickers == tickers


def test_constructor_starts_with_no_trained_models() -> None:
    """Check that no model is registered right after construction."""
    manager = _build_manager()

    assert not manager.models


def test_default_trainer_factory_is_the_base_autoencoder() -> None:
    """Check that the default factory is the base autoencoder builder."""
    manager = _build_manager()

    assert manager.trainer_factory is build_trainer


def test_list_tickers_keeps_the_constructor_order() -> None:
    """Check that the tickers are listed in the order they were given."""
    manager = _build_manager()

    assert manager.list_tickers() == ["AAPL", "MSFT"]


def test_get_date_range_returns_the_configured_dates() -> None:
    """Check that the start and end dates of a ticker are returned."""
    manager = _build_manager()

    assert manager.get_date_range("MSFT") == ("2021-01-01", "2026-01-01")


def test_get_date_range_rejects_an_unknown_ticker() -> None:
    """Check that an unknown ticker raises KeyError."""
    manager = _build_manager()

    with pytest.raises(KeyError):
        manager.get_date_range("PETR4.SA")


def test_fetch_price_data_uses_the_ticker_dates() -> None:
    """Check that the prices are built for the dates of the ticker."""
    manager = _build_manager()
    expected_dataset = pd.DataFrame({"close_price": [1.0]})

    with patch(
        BUILD_DATASET_PATH, return_value=expected_dataset
    ) as mock_build_dataset:
        dataset = manager.fetch_price_data("AAPL")

    assert dataset is expected_dataset
    assert mock_build_dataset.call_args.args[:3] == (
        "AAPL",
        "2020-01-01",
        "2026-01-01",
    )


def test_train_model_creates_trains_and_stores_the_result() -> None:
    """Check that the trainer of the ticker runs and its result is kept."""
    factory = _make_factory()
    manager = _build_manager(trainer_factory=factory)
    price_data = pd.DataFrame({"close_price": [1.0]})

    with patch(BUILD_DATASET_PATH, return_value=price_data):
        result = manager.train_model("AAPL")

    factory.assert_called_once_with("AAPL", price_data)
    assert result is factory.return_value.train.return_value
    assert manager.get_model("AAPL") is result


def test_train_all_models_trains_every_ticker_in_order() -> None:
    """Check that train_model runs once per ticker, in listed order."""
    manager = _build_manager()

    with patch.object(
        TickerModelManager, "list_tickers", return_value=["MSFT", "AAPL"]
    ), patch.object(TickerModelManager, "train_model") as mock_train_model:
        manager.train_all_models()

    assert [call.args[0] for call in mock_train_model.call_args_list] == [
        "MSFT",
        "AAPL",
    ]


def test_train_all_models_does_nothing_without_tickers() -> None:
    """Check that no training happens when there are no tickers."""
    manager = TickerModelManager(tickers={})

    with patch.object(
        TickerModelManager, "list_tickers", return_value=[]
    ), patch.object(TickerModelManager, "train_model") as mock_train_model:
        manager.train_all_models()

    mock_train_model.assert_not_called()


def test_train_all_models_trains_sequentially_not_in_parallel() -> None:
    """Check that a ticker only starts training after the previous ends."""
    manager = _build_manager()
    events = []

    def fake_train_model(ticker: str) -> None:
        events.append(f"start {ticker}")
        events.append(f"end {ticker}")

    with patch.object(
        TickerModelManager, "list_tickers", return_value=["AAPL", "MSFT"]
    ), patch.object(
        TickerModelManager, "train_model", side_effect=fake_train_model
    ):
        manager.train_all_models()

    assert events == ["start AAPL", "end AAPL", "start MSFT", "end MSFT"]


def test_train_all_models_stops_at_first_failure() -> None:
    """Check that an error propagates and later tickers are not trained."""
    manager = _build_manager()

    with patch.object(
        TickerModelManager, "list_tickers", return_value=["AAPL", "MSFT"]
    ), patch.object(
        TickerModelManager,
        "train_model",
        side_effect=RuntimeError("training failed"),
    ) as mock_train_model:
        with pytest.raises(RuntimeError, match="training failed"):
            manager.train_all_models()

    mock_train_model.assert_called_once_with("AAPL")


def test_get_model_rejects_a_ticker_not_trained_yet() -> None:
    """Check that asking for an untrained model raises KeyError."""
    manager = _build_manager()

    with pytest.raises(KeyError, match="train_model"):
        manager.get_model("AAPL")
